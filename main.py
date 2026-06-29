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

from flask import Flask, Response, jsonify, render_template, request
import database as db

# =====================================================================
# LOAD CONFIGURATION
# =====================================================================
with open("configs/default.yaml", "r") as f:
    CFG = yaml.safe_load(f)

db.init_db(CFG["database"]["path"])

# =====================================================================
# PLUGIN LOADER
# =====================================================================
_plugins     = {}   # { use_case_name: module }
_uc_states   = {}   # { use_case_name: state_dict }
_active_ucs  = set()

def load_plugin(name):
    """Import plugins/<name>.py, call init(), store fresh state."""
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

# Pre-load all enabled use cases
for uc_name, uc_cfg in CFG["use_cases"].items():
    if uc_cfg.get("enabled", False):
        load_plugin(uc_name)
        _active_ucs.add(uc_name)

# =====================================================================
# SHARED STATE
# =====================================================================
_latest_frame  = None
_frame_lock    = threading.Lock()
_alarm_active  = False
_simulating    = False
_sim_end_time  = 0
_frame_counter = 0

# =====================================================================
# ALARM (pyttsx3 background thread)
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
# PREPROCESSING
# =====================================================================
def preprocess(cap):
    global _frame_counter
    source = CFG["camera"]["source"]
    is_live = isinstance(source, int)

    if is_live:
        for _ in range(3):
            cap.grab()
        ret, frame = cap.retrieve()
    else:
        ret, frame = cap.read()

    if not ret or frame is None:
        return None, None, False

    _frame_counter += 1
    run = (_frame_counter % CFG["camera"]["process_every_n"] == 0)

    if run:
        w, h = CFG["camera"]["infer_size"]
        return cv2.resize(frame, (w, h)), frame, True
    return None, frame, False

# =====================================================================
# PROCESSING  — runs all active plugins
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
                display_frame, infer_frame, state
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
# POSTPROCESSING — HUD drawn onto display_frame
# =====================================================================
DARK_NAVY = (53,  27,  15)
DARK_BLUE = (58,  40,  20)
GREEN     = (50,  205, 50)
AMBER     = (0,   165, 255)
RED       = (60,  60,  220)
WHITE     = (255, 255, 255)
YELLOW    = (0,   210, 240)
GRAY      = (160, 160, 160)

STATUS_COLOR = {
    "COMPLIANT": GREEN, "WARNING": AMBER,
    "ALERT": RED,       "IDLE": GRAY,
}

def _alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.75):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# =====================================================================
# PHASE 3 — POSTPROCESSING
# Only draws what the HTML dashboard cannot show:
#   - Bounding boxes (person + PPE)
#   - Status word overlay on video (UNSAFE / WARNING)
#   - LIVE / CAM label + REC indicator
# Everything else (sidebar, pills, cards, log) is in dashboard.html
# =====================================================================

def postprocess(frame):
    from datetime import datetime
    H, W = frame.shape[:2]

    # Pick primary use case status
    primary = next(iter(_active_ucs), None)
    p_state = _uc_states.get(primary, {}) if primary else {}
    status  = p_state.get("status", "IDLE")
    missing = p_state.get("missing", [])
    color   = STATUS_COLOR.get(status, GRAY)

    # ── LIVE label (top-left) ─────────────────────────────────────
    cv2.circle(frame, (14, 14), 5, RED, -1)
    cv2.putText(frame, "LIVE  CAM 1",
                (24, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, WHITE, 1, cv2.LINE_AA)

    # ── REC + timestamp (top-right) ───────────────────────────────
    dt = datetime.now().strftime("%d/%m/%Y  %I:%M%p")
    (tw, _), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
    cv2.putText(frame, dt, (W - tw - 8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (W - 38, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)

    # ── Big status word on video (WARNING / UNSAFE only) ──────────
    if status in ("ALERT", "WARNING"):
        big = "UNSAFE" if status == "ALERT" else "WARNING"
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        # Semi-transparent backing so text is readable on any background
        _alpha_rect(frame, W - tw - 20, H//2 - th - 16,
                    W - 8, H//2 + 10, (0, 0, 0), 0.45)
        cv2.putText(frame, big, (W - tw - 14, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

    # ── Timecode (bottom-centre) ──────────────────────────────────
    tc = datetime.now().strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(tc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame, tc, ((W - tw)//2, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

    return frame

# =====================================================================
# CAMERA LOOP (background thread)
# =====================================================================
def camera_loop():
    source = CFG["camera"]["source"]
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Camera] Cannot open source: {source}")
        return
    print(f"[Camera] Stream started — source: {source}")

    while True:
        infer_frame, display_frame, run_infer = preprocess(cap)
        if display_frame is None:
            time.sleep(0.01)
            continue

        if run_infer and _active_ucs:
            display_frame = process(infer_frame, display_frame)

        display_frame = postprocess(display_frame)

        with _frame_lock:
            global _latest_frame
            _latest_frame = display_frame.copy()

    cap.release()


# =====================================================================
# FLASK APP
# =====================================================================
app = Flask(__name__, template_folder="templates")

def _mjpeg_generator():
    quality = CFG["camera"]["mjpeg_quality"]
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.033)
            continue

        # Resize to fill the dashboard video area (matches CSS flex: 1)
        # Target 16:9 at 1280×720 — browser scales to fit the panel
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        _, buf = cv2.imencode(".jpg", frame,
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)

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
    _sim_end_time = time.time() + 5.0   # 5-second simulation
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
    # Start camera in background
    threading.Thread(target=camera_loop, daemon=True).start()
 
    host = CFG["server"]["host"]
    port = CFG["server"]["port"]
    print(f"\n=== SIDP Dashboard → http://localhost:{port} ===\n")
    app.run(host=host, port=port, debug=False, use_reloader=False,
            threaded=True)