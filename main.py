"""
SIDP — Smart Integrated Detection Platform
Single entry point: python main.py
No cv2.imshow / cv2.namedWindow / cv2.waitKey (NFR6)
"""
import cv2, time, threading, importlib, yaml, os
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request, send_file
import database as db
import alarm

with open("configs/default.yaml", "r") as f:
    CFG = yaml.safe_load(f)
db.init_db(CFG["database"]["path"])
os.makedirs("data/recordings", exist_ok=True)
SOURCE  = CFG["camera"]["source"]
IS_LIVE = isinstance(SOURCE, int)

_plugins = {}; _uc_states = {}; _active_ucs = set()
_last_alert_notify = {}   # { use_case: last notify time }
NOTIFY_COOLDOWN    = 10.0  # seconds between audio alerts per use case

def load_plugin(name):
    if name in _plugins: return
    try:
        mod = importlib.import_module(f"plugins.{name}")
        mod.init(CFG["use_cases"][name])
        _plugins[name] = mod; _uc_states[name] = mod.fresh_state()
        print(f"[Loader] Plugin '{name}' ready")
    except Exception as e:
        print(f"[Loader] ERROR loading plugin '{name}': {e}")

for uc_name, uc_cfg in CFG["use_cases"].items():
    if uc_cfg.get("enabled", False):
        load_plugin(uc_name); _active_ucs.add(uc_name)

_latest_frame   = None; _frame_lock   = threading.Lock()
_raw_frame      = None; _raw_lock     = threading.Lock()
_inferred_frame = None; _infer_lock   = threading.Lock()
_simulating     = False; _sim_end_time = 0
_frame_counter  = 0;    _use_person   = True

# =====================================================================
# VIOLATION RECORDER
# =====================================================================
_recorder = None; _recorder_lock = threading.Lock()
_recorder_frames = 0; _recorder_max = 150; _recorder_event_id = None

def _start_recording(event_id):
    global _recorder, _recorder_frames, _recorder_event_id
    with _recorder_lock:
        if _recorder is not None: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/recordings/alert_{ts}.mp4"
        for fc in ["avc1","mp4v","XVID"]:
            w = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*fc), 20, (1280,720))
            if w.isOpened(): print(f"[Recorder] Codec: {fc}"); break
            w.release()
        else: print("[Recorder] No codec found"); return
        _recorder = w; _recorder_frames = 0
        _recorder_event_id = (event_id, filename)
        print(f"[Recorder] Started → {filename} | event_id={event_id}")

def _write_recording_frame(frame):
    global _recorder, _recorder_frames, _recorder_event_id
    saved = None
    with _recorder_lock:
        if _recorder is None: return
        _recorder.write(cv2.resize(frame,(1280,720))); _recorder_frames += 1
        if _recorder_frames >= _recorder_max:
            _recorder.release(); _recorder = None
            if _recorder_event_id: saved = _recorder_event_id; _recorder_event_id = None
    if saved:
        db.update_clip_path(saved[0], saved[1])
        print(f"[Recorder] Auto-saved → {saved[1]} | event_id={saved[0]}")

def _stop_recording():
    global _recorder, _recorder_event_id
    saved = None
    with _recorder_lock:
        if _recorder is None: return
        _recorder.release(); _recorder = None
        if _recorder_event_id: saved = _recorder_event_id; _recorder_event_id = None
    if saved:
        db.update_clip_path(saved[0], saved[1])
        print(f"[Recorder] Stopped → {saved[1]} | event_id={saved[0]}")

# =====================================================================
# PROCESSING
# =====================================================================
def process(infer_frame, display_frame):
    global _simulating, _sim_end_time
    any_alert   = False
    alert_set   = set()   # use cases currently in ALERT this frame
    alert_set   = {n for n in _active_ucs if _uc_states.get(n,{}).get("status") == "ALERT"}
    warning_set = {n for n in _active_ucs if _uc_states.get(n,{}).get("status") == "WARNING"}

    # First pass — run all plugins, collect statuses
    for name in list(_active_ucs):
        plugin = _plugins.get(name); state = _uc_states.get(name)
        if plugin is None or state is None: continue
        try:
            display_frame, new_state = plugin.process_frame(display_frame, infer_frame, state)
            _uc_states[name] = new_state

            status = new_state["status"]
            if status == "ALERT":
                any_alert = True
                alert_set.add(name)
                # Start recording on first ALERT frame per episode
                if not new_state.get("recording_started"):
                    eid = new_state.get("alert_event_id")
                    if eid:
                        _start_recording(eid)
                        _uc_states[name]["recording_started"] = True
                        print(f"[Engine] Recording for event_id={eid}")

        except Exception as e:
            print(f"[Engine] Plugin '{name}' error: {e}")

    # Simulation override
    if _simulating:
        if time.time() < _sim_end_time:
            any_alert = True
            for n in _active_ucs:
                if n in _uc_states:
                    _uc_states[n]["status"]  = "ALERT"
                    _uc_states[n]["missing"] = ["[Simulated violation]"]
                    alert_set.add(n)
        else:
            _simulating = False

    # Then pass both to notify
    now = time.time()

    # Get shortest cooldown among all currently violating use cases
    violating = alert_set | warning_set
    combined_cooldown = min(
        CFG["use_cases"].get(n, {}).get("alert_cooldown", NOTIFY_COOLDOWN)
        for n in violating
    ) if violating else NOTIFY_COOLDOWN


    for name in list(_active_ucs):
        st     = _uc_states.get(name, {})
        status = st.get("status", "IDLE")
        if status in ("ALERT", "WARNING"):
            last     = _last_alert_notify.get(name, 0)
            cooldown = CFG["use_cases"].get(name, {}).get("alert_cooldown", NOTIFY_COOLDOWN)
            if now - last >= cooldown:
                alarm.notify(name, status, alert_set, warning_set, combined_cooldown)
                _last_alert_notify[name] = now
        else:
            _last_alert_notify.pop(name, None)

    if any_alert:
        pass   # alarm.notify() handles speech — no global start needed
    else:
        _stop_recording()
        for n in _active_ucs:
            if n in _uc_states:
                _uc_states[n]["recording_started"] = False
                _uc_states[n]["alert_event_id"]    = None

    return display_frame

# =====================================================================
# POSTPROCESSING CONSTANTS
# =====================================================================
RED   = (60,  60,  220)
GREEN = (50,  205, 50)
AMBER = (0,   165, 255)
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)
BLACK = (0,   0,   0)

STATUS_COLOR = {"COMPLIANT":GREEN,"WARNING":AMBER,"ALERT":RED,"IDLE":GRAY}
UC_LABELS    = {"ppe":"PPE","occupancy":"OCCUPANCY","no_phone":"NO PHONE"}
STATUS_LABEL = {"COMPLIANT":"SAFE","WARNING":"WARNING","ALERT":"UNSAFE","IDLE":"IDLE"}

def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.6):
    ov = frame.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

# =====================================================================
# POSTPROCESSING
# =====================================================================
def postprocess(frame):
    H, W = frame.shape[:2]

    STREAM_W      = 1280
    RIGHT_PANEL_W = 288
    scale_x       = W / STREAM_W
    VISIBLE_RIGHT = int((STREAM_W - RIGHT_PANEL_W) * scale_x)
    VISIBLE_LEFT  = 0

    # LIVE label
    cv2.circle(frame, (14,14), 5, RED, -1)
    cv2.putText(frame, "LIVE  CAM 1", (24,19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

    # Timestamp
    dt = datetime.now().strftime("%d/%m/%Y  %I:%M%p")
    (tw,_),_ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(frame, dt, (VISIBLE_RIGHT-tw-8,14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, WHITE, 1, cv2.LINE_AA)

    # REC indicator
    with _recorder_lock: recording = _recorder is not None
    if recording:
        cv2.circle(frame, (VISIBLE_RIGHT-52,30), 5, (0,0,255), -1)
        cv2.putText(frame, "REC", (VISIBLE_RIGHT-42,34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "REC", (VISIBLE_RIGHT-38,34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

    # Horizontal pill badges centred at top
    BADGE_H = 28; BADGE_PAD = 10; GAP = 8; BADGE_Y = 35

    violations = [
        n for n in ["ppe","occupancy","no_phone"]
        if n in _active_ucs
        and _uc_states.get(n,{}).get("status") not in ("IDLE","COMPLIANT")
    ]

    if violations:
        widths = []
        for name in violations:
            st  = _uc_states.get(name,{})
            st_ = st.get("status","IDLE")
            ul  = UC_LABELS.get(name, name.upper())
            sl  = STATUS_LABEL.get(st_, st_)
            cd  = st.get("countdown",0)
            cdt = f"{cd:.1f}s" if st_=="WARNING" and cd>0 else ""
            (uw,_),_ = cv2.getTextSize(ul,  cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            (sw,_),_ = cv2.getTextSize(sl,  cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            (cw,_),_ = cv2.getTextSize(cdt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            content  = 8+6+uw+8+sw+(6+cw if cdt else 0)
            widths.append(content+BADGE_PAD*2)

        total_w = sum(widths)+GAP*(len(widths)-1)
        cx      = (VISIBLE_LEFT+VISIBLE_RIGHT)//2 + 50
        bx      = cx - total_w//2

        for i, name in enumerate(violations):
            st  = _uc_states.get(name,{})
            st_ = st.get("status","IDLE")
            col = STATUS_COLOR.get(st_, GRAY)
            ul  = UC_LABELS.get(name, name.upper())
            sl  = STATUS_LABEL.get(st_, st_)
            cd  = st.get("countdown",0)
            cdt = f"{cd:.1f}s" if st_=="WARNING" and cd>0 else ""
            x   = bx + sum(widths[:i]) + GAP*i
            bw  = widths[i]

            _alpha_rect(frame, x, BADGE_Y, x+bw, BADGE_Y+BADGE_H, BLACK, 0.65)
            cv2.rectangle(frame,(x,BADGE_Y),(x+bw,BADGE_Y+BADGE_H),col,1)
            dot_x = x+BADGE_PAD; dot_y = BADGE_Y+BADGE_H//2
            cv2.circle(frame,(dot_x,dot_y),4,col,-1)
            tx = dot_x+14
            cv2.putText(frame,ul,(tx,BADGE_Y+19),cv2.FONT_HERSHEY_SIMPLEX,0.38,WHITE,1,cv2.LINE_AA)
            (uw,_),_ = cv2.getTextSize(ul,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
            sx = tx+uw+8
            cv2.putText(frame,sl,(sx,BADGE_Y+19),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)
            if cdt:
                (sw,_),_ = cv2.getTextSize(sl,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
                cv2.putText(frame,cdt,(sx+sw+6,BADGE_Y+19),cv2.FONT_HERSHEY_SIMPLEX,0.36,AMBER,1,cv2.LINE_AA)

    # Timecode bottom-centre
    tc = datetime.now().strftime("%H:%M:%S")
    (tw,_),_ = cv2.getTextSize(tc,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
    vcx = (VISIBLE_LEFT+VISIBLE_RIGHT)//2
    cv2.putText(frame,tc,(vcx-tw//2,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.4,WHITE,1,cv2.LINE_AA)
    return frame

# =====================================================================
# THREADS
# =====================================================================
def capture_loop():
    global _raw_frame
    cap = cv2.VideoCapture(SOURCE); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened(): print(f"[Camera] Cannot open: {SOURCE}"); return
    print(f"[Camera] Capture started — source: {SOURCE}")
    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.01); continue
        with _raw_lock: _raw_frame = frame.copy()
    cap.release()

def inference_loop():
    global _frame_counter, _inferred_frame, _use_person
    print("[Inference] Thread started")
    while True:
        with _raw_lock: frame = _raw_frame
        if frame is None: time.sleep(0.005); continue
        _frame_counter += 1
        w,h = CFG["camera"]["infer_size"]
        infer = cv2.resize(frame,(w,h)); display = frame.copy()
        if _active_ucs: display = process(infer,display); _use_person = not _use_person
        with _infer_lock: _inferred_frame = display.copy()

def display_loop():
    global _latest_frame
    print("[Display] Thread started"); target = 1.0/30
    while True:
        t0 = time.time()
        with _infer_lock: frame = _inferred_frame
        if frame is None:
            with _raw_lock: frame = _raw_frame
        if frame is None: time.sleep(0.01); continue
        display = postprocess(frame.copy())
        _write_recording_frame(display)
        with _frame_lock: _latest_frame = display
        leftover = target-(time.time()-t0)
        if leftover>0: time.sleep(leftover)

# =====================================================================
# FLASK
# =====================================================================
app = Flask(__name__, template_folder="templates")

def _mjpeg_generator():
    quality = CFG["camera"]["mjpeg_quality"]; target = 1.0/30
    while True:
        t0 = time.time()
        with _frame_lock: frame = _latest_frame
        if frame is None: time.sleep(0.01); continue
        frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_LINEAR)
        _,buf = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,quality])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        leftover = target-(time.time()-t0)
        if leftover>0: time.sleep(leftover)

@app.route("/")
def index(): return render_template("dashboard.html")

@app.route("/video")
def video(): return Response(_mjpeg_generator(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    out = {n:{"status":s.get("status","IDLE"),"missing":s.get("missing",[]),"detected":s.get("detected",[]),"countdown":s.get("countdown",0)} for n,s in _uc_states.items()}
    return jsonify({"states":out,"active":list(_active_ucs)})

@app.route("/api/summary")
def api_summary(): return jsonify(db.get_summary(request.args.getlist("uc") or None))

@app.route("/api/events")
def api_events(): return jsonify(db.get_recent(50,request.args.getlist("uc") or None))

@app.route("/api/history")
def api_history(): return jsonify(db.get_recent(1000,request.args.getlist("uc") or None))

@app.route("/api/activate", methods=["POST"])
def api_activate():
    data=request.json; name=data.get("name"); on=data.get("active",True)
    if name not in _plugins: load_plugin(name)
    if on:
        _active_ucs.add(name)
        if name not in _uc_states: _uc_states[name]=_plugins[name].fresh_state()
    else: _active_ucs.discard(name)
    return jsonify({"ok":True,"active":list(_active_ucs)})

@app.route("/api/silence", methods=["POST"])
def api_silence():
    alarm.silence()
    return jsonify({"ok":True})

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    global _simulating,_sim_end_time
    _simulating=True; _sim_end_time=time.time()+5.0
    return jsonify({"ok":True})

@app.route("/api/clear_log", methods=["POST"])
def api_clear_log(): db.clear_events(request.json.get("use_cases") or None); return jsonify({"ok":True})

@app.route("/api/clip/<int:event_id>")
def api_clip(event_id):
    row = next((r for r in db.get_recent(1000) if r["id"]==event_id),None)
    if not row or not row.get("clip_path"): return jsonify({"error":"No clip"}),404
    if not os.path.exists(row["clip_path"]): return jsonify({"error":"File missing"}),404
    return send_file(row["clip_path"],mimetype="video/mp4")

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    alarm.start()   # start speaker thread before camera
    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=display_loop,   daemon=True).start()

    host=CFG["server"]["host"]; port=CFG["server"]["port"]
    print(f"\n=== SIDP Dashboard → http://localhost:{port} ===\n")
    app.run(host=host,port=port,debug=False,use_reloader=False,threaded=True)
