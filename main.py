"""
SIDP — Smart Integrated Detection Platform
Single entry point: python main.py
No cv2.imshow / cv2.namedWindow / cv2.waitKey (NFR6)
"""
import cv2
import time
import threading
import importlib
import yaml
import pyttsx3
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request
import database as db

# =====================================================================
# LOAD CONFIGURATION
# =====================================================================
with open("configs/default.yaml", "r") as f:
    CFG = yaml.safe_load(f)

db.init_db(CFG["database"]["path"])

SOURCE  = CFG["camera"]["source"]
IS_LIVE = isinstance(SOURCE, int)

# =====================================================================
# PLUGIN LOADER
# =====================================================================
_plugins    = {}
_uc_states  = {}
_active_ucs = set()

def load_plugin(name):
    if name in _plugins:
        return
    try:
        mod = importlib.import_module(f"plugins.{name}")
        mod.init(CFG["use_cases"][name])
        _plugins[name]   = mod
        _uc_states[name] = mod.fresh_state()
        print(f"[Loader] Plugin '{name}' ready")
    except Exception as e:
        print(f"[Loader] ERROR loading plugin '{name}': {e}")

for uc_name, uc_cfg in CFG["use_cases"].items():
    if uc_cfg.get("enabled", False):
        load_plugin(uc_name)
        _active_ucs.add(uc_name)

# =====================================================================
# SHARED STATE
# =====================================================================
_latest_frame   = None   # final display frame → MJPEG
_frame_lock     = threading.Lock()

_raw_frame      = None   # latest raw frame from camera
_raw_lock       = threading.Lock()

_inferred_frame = None   # latest frame with boxes drawn
_infer_lock     = threading.Lock()

_alarm_active   = False
_simulating     = False
_sim_end_time   = 0
_frame_counter  = 0
_use_person     = True   # alternates per frame for model switching

# =====================================================================
# ALARM
# =====================================================================
def _alarm_loop():
    while _alarm_active:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.say("Critical safety breach. Equip your PPE immediately.")
            engine.runAndWait()
        except Exception:
            pass
        time.sleep(3.5)

def _start_alarm():
    global _alarm_active
    if not _alarm_active:
        _alarm_active = True
        threading.Thread(target=_alarm_loop, daemon=True).start()

def _stop_alarm():
    global _alarm_active
    _alarm_active = False

# =====================================================================
# PROCESSING — runs all active plugins
# =====================================================================
def process(infer_frame, display_frame):
    global _simulating, _sim_end_time
    any_alert = False

    for name in list(_active_ucs):
        plugin = _plugins.get(name)
        state  = _uc_states.get(name)
        if plugin is None or state is None:
            continue
        try:
            display_frame, new_state = plugin.process_frame(
                display_frame,
                infer_frame,
                state
            )
            _uc_states[name] = new_state
            if new_state["status"] == "ALERT":
                any_alert = True
        except Exception as e:
            print(f"[Engine] Plugin '{name}' error: {e}")

    # Simulation override
    if _simulating:
        if time.time() < _sim_end_time:
            any_alert = True
            for name in _active_ucs:
                if name in _uc_states:
                    _uc_states[name]["status"]  = "ALERT"
                    _uc_states[name]["missing"] = ["[Simulated violation]"]
        else:
            _simulating = False

    if any_alert:
        _start_alarm()
    else:
        _stop_alarm()

    return display_frame

# =====================================================================
# POSTPROCESSING
# Only draws what HTML cannot: bounding boxes, status word, LIVE/REC
# =====================================================================
RED   = (60,  60,  220)
GREEN = (50,  205, 50)
AMBER = (0,   165, 255)
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)

STATUS_COLOR = {
    "COMPLIANT": GREEN, "WARNING": AMBER,
    "ALERT": RED,       "IDLE": GRAY,
}

def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def postprocess(frame):
    H, W = frame.shape[:2]

    # Determine the HIGHEST SEVERITY status across ALL active use cases
    # Priority: ALERT > WARNING > COMPLIANT > IDLE
    SEVERITY = {"ALERT": 3, "WARNING": 2, "COMPLIANT": 1, "IDLE": 0}

    worst_status  = "IDLE"
    worst_missing = []

    for name in _active_ucs:
        st = _uc_states.get(name, {})
        s  = st.get("status", "IDLE")
        if SEVERITY.get(s, 0) > SEVERITY.get(worst_status, 0):
            worst_status  = s
            worst_missing = st.get("missing", [])

    color = STATUS_COLOR.get(worst_status, GRAY)

    # LIVE label (top-left)
    cv2.circle(frame, (14, 14), 5, RED, -1)
    cv2.putText(frame, "LIVE  CAM 1",
                (24, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, WHITE, 1, cv2.LINE_AA)

    # Timestamp + REC (top-right)
    dt = datetime.now().strftime("%d/%m/%Y  %I:%M%p")
    (tw, _), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(frame, dt, (W - tw - 8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (W - 38, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)

    # Big status word on video — reflects worst status across all active UCs
    if worst_status in ("ALERT", "WARNING"):
        big = "UNSAFE" if worst_status == "ALERT" else "WARNING"
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        _alpha_rect(frame, W - tw - 20, H//2 - th - 16,
                    W - 8, H//2 + 10, (0, 0, 0), 0.45)
        cv2.putText(frame, big, (W - tw - 14, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

    # Timecode (bottom-centre)
    tc = datetime.now().strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(tc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame, tc, ((W - tw)//2, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

    return frame

# =====================================================================
# THREAD 1 — CAPTURE
# Reads frames as fast as possible, never blocked by inference
# =====================================================================
def capture_loop():
    global _raw_frame

    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[Camera] Cannot open source: {SOURCE}")
        return
    print(f"[Camera] Capture started — source: {SOURCE}")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with _raw_lock:
            _raw_frame = frame.copy()

    cap.release()

# =====================================================================
# THREAD 2 — INFERENCE
# Runs YOLO as fast as GPU allows, alternates models each frame
# =====================================================================
def inference_loop():
    global _frame_counter, _inferred_frame, _use_person

    print("[Inference] Thread started")

    while True:
        with _raw_lock:
            frame = _raw_frame
        if frame is None:
            time.sleep(0.005)
            continue

        _frame_counter += 1
        w, h        = CFG["camera"]["infer_size"]
        infer_frame = cv2.resize(frame, (w, h))
        display     = frame.copy()

        if _active_ucs:
            # Alternate: person model on odd frames, PPE model on even frames
            display = process(infer_frame, display)
            _use_person = not _use_person

        with _infer_lock:
            _inferred_frame = display.copy()
        # No sleep — run as fast as GPU allows

# =====================================================================
# THREAD 3 — DISPLAY
# Composites HUD at steady 30fps, never waits for inference
# =====================================================================
def display_loop():
    global _latest_frame

    print("[Display] Thread started")
    target = 1.0 / 30   # 30fps target

    while True:
        t0 = time.time()

        # Prefer inferred frame, fall back to raw if not ready
        with _infer_lock:
            frame = _inferred_frame
        if frame is None:
            with _raw_lock:
                frame = _raw_frame
        if frame is None:
            time.sleep(0.01)
            continue

        display = postprocess(frame.copy())

        with _frame_lock:
            _latest_frame = display

        elapsed  = time.time() - t0
        leftover = target - elapsed
        if leftover > 0:
            time.sleep(leftover)

# =====================================================================
# FLASK APP
# =====================================================================
app = Flask(__name__, template_folder="templates")

def _mjpeg_generator():
    quality = CFG["camera"]["mjpeg_quality"]
    target  = 1.0 / 30

    while True:
        t0 = time.time()

        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

        elapsed  = time.time() - t0
        leftover = target - elapsed
        if leftover > 0:
            time.sleep(leftover)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/video")
def video():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    out = {}
    for name, state in _uc_states.items():
        out[name] = {
            "status":    state.get("status", "IDLE"),
            "missing":   state.get("missing", []),
            "detected":  state.get("detected", []),
            "countdown": state.get("countdown", 0),
        }
    # Returns ALL use case states — frontend decides how many to render
    return jsonify({"states": out, "active": list(_active_ucs)})

@app.route("/api/summary")
def api_summary():
    ucs = request.args.getlist("uc") or None
    return jsonify(db.get_summary(ucs))

@app.route("/api/events")
def api_events():
    ucs = request.args.getlist("uc") or None
    return jsonify(db.get_recent(50, ucs))

@app.route("/api/activate", methods=["POST"])
def api_activate():
    data = request.json
    name = data.get("name")
    on   = data.get("active", True)
    if name not in _plugins:
        load_plugin(name)
    if on:
        _active_ucs.add(name)
        if name not in _uc_states:
            _uc_states[name] = _plugins[name].fresh_state()
    else:
        _active_ucs.discard(name)
    return jsonify({"ok": True, "active": list(_active_ucs)})

@app.route("/api/silence", methods=["POST"])
def api_silence():
    _stop_alarm()
    return jsonify({"ok": True})

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    global _simulating, _sim_end_time
    _simulating   = True
    _sim_end_time = time.time() + 5.0
    return jsonify({"ok": True})

@app.route("/api/clear_log", methods=["POST"])
def api_clear_log():
    ucs = request.json.get("use_cases") or None
    db.clear_events(ucs)
    return jsonify({"ok": True})

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    threading.Thread(target=capture_loop,   daemon=True).start()  # Thread 1
    threading.Thread(target=inference_loop, daemon=True).start()  # Thread 2
    threading.Thread(target=display_loop,   daemon=True).start()  # Thread 3

    host = CFG["server"]["host"]
    port = CFG["server"]["port"]
    print(f"\n=== SIDP Dashboard → http://localhost:{port} ===\n")
    app.run(host=host, port=port, debug=False,
            use_reloader=False, threaded=True)
