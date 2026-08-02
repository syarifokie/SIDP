"""
Standalone Safety Mask Detection Monitor
Self-contained version of the original plugin (init/process_frame/fresh_state)
so it can be run directly with: python safety_mask_standalone.py

Replaces the external `database.insert_event` call with a simple CSV logger,
and replaces the external main-loop/config injection with a built-in webcam
loop and CONFIG dict.

NOTE: This plugin uses TWO models:
  1. model_path        -> a custom-trained PPE model with a "mask" class
                           (e.g. ppe.pt from your earlier PPE script)
  2. person_model_path  -> a generic YOLO model with the standard COCO
                           "person" class (e.g. yolov8n.pt)
Both files must exist locally, or ultralytics will try to download them
(which failed for you earlier due to a bad proxy env var — see below).
"""
import cv2
import time
import os
import csv
from datetime import datetime
import pyttsx3
import threading

# =====================================================================
# CONFIG  (previously passed in via init(config) from the host app)
# =====================================================================
CONFIG = {
    "model_path": os.path.join("model", "ppe.pt"),   # custom PPE model (must contain "mask" class)
    "person_model_path": "yolov8n.pt",                # generic person detector
    "conf_threshold": 0.40,
    "person_conf_threshold": 0.30,
    "mask_class": "mask",
    "grace_period": 3.0,
}

ALERT_COOLDOWN = 6  # seconds between repeated voice alerts

LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "safety_mask_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Type", "Status", "Missing", "Detected"])

GREEN = (50,  205, 50)
RED   = (60,  60,  220)
CYAN  = (255, 220, 0)
WHITE = (255, 255, 255)

_cfg          = {}
_mask_model   = None
_person_model = None


# =====================================================================
# BACKGROUND VOICE ENGINE
# =====================================================================
def _speak_worker(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def trigger_voice_alert(text):
    threading.Thread(
        target=_speak_worker,
        args=(text,),
        daemon=True
    ).start()


# =====================================================================
# Local replacement for database.insert_event
# =====================================================================
def insert_event(event_type, status, missing, detected):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="") as f:
        csv.writer(f).writerow([timestamp, event_type, status, "; ".join(missing), "; ".join(detected)])
    print(f"[{timestamp}] {event_type.upper()} EVENT: {status} | missing={missing} | detected={detected}")
    return timestamp  # stand-in for a real event_id


# =====================================================================
# init / fresh_state / process_frame  (unchanged from original plugin)
# =====================================================================
def init(config):
    global _cfg, _mask_model, _person_model
    _cfg = config
    from ultralytics import YOLO
    _mask_model   = YOLO(config["model_path"])
    _person_model = YOLO(config["person_model_path"])
    print(f"[SafetyMask] Models loaded — filtering class: {config.get('mask_class','mask')}")


def fresh_state():
    return {
        "status":            "IDLE",
        "violation_start":   None,
        "staff_notified":    False,
        "countdown":         _cfg.get("grace_period", 3.0),
        "missing":           [],
        "detected":          [],
        "last_person_boxes": [],
        "person_miss_count": 0,
        "person_miss_limit": 5,
        "recording_started": False,
        "alert_event_id":    None,
    }


def _has_mask_near_person(person_box, mask_boxes, threshold=0.3):
    px1, py1, px2, py2 = person_box
    upper_y2 = py1 + (py2 - py1) * 0.5

    for mx1, my1, mx2, my2 in mask_boxes:
        ix1 = max(px1, mx1); iy1 = max(py1, my1)
        ix2 = min(px2, mx2); iy2 = min(upper_y2, my2)
        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            mask_area    = (mx2 - mx1) * (my2 - my1)
            if mask_area > 0 and (intersection / mask_area) >= threshold:
                return True
    return False


def process_frame(display_frame, infer_frame, state, person_results=None, run_person=True, run_ppe=True):
    conf        = _cfg.get("conf_threshold", 0.40)
    person_conf = _cfg.get("person_conf_threshold", 0.30)
    mask_class  = _cfg.get("mask_class", "mask")
    grace       = _cfg.get("grace_period", 3.0)

    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    # ── Person detection ─────────────────────────────────────────
    if run_person:
        p_results = _person_model(infer_frame, verbose=False, conf=person_conf, classes=[0])
        if len(p_results[0].boxes) > 0:
            state["last_person_boxes"] = p_results[0].boxes
            state["person_miss_count"] = 0
        else:
            state["person_miss_count"] += 1

    person_found = state["person_miss_count"] < state["person_miss_limit"]

    # ── Mask detection — per-person pairing ──────────────────────
    violators = 0
    mask_boxes = []

    if run_ppe:
        ppe_results = _mask_model(infer_frame, verbose=False, conf=conf)
        detected    = set()

        for box in ppe_results[0].boxes:
            cls_name = _mask_model.names[int(box.cls[0])]
            detected.add(cls_name)
            if cls_name == mask_class:
                mask_boxes.append(tuple(map(int, box.xyxy[0])))

        state["detected"]   = list(detected)
        state["mask_boxes"] = mask_boxes
    else:
        mask_boxes = state.get("mask_boxes", [])

    for box in state["last_person_boxes"]:
        px1, py1, px2, py2 = map(int, box.xyxy[0])
        dx1, dy1 = int(px1 * sx), int(py1 * sy)
        dx2, dy2 = int(px2 * sx), int(py2 * sy)

        compliant = _has_mask_near_person((px1, py1, px2, py2), mask_boxes)
        color     = GREEN if compliant else RED

        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
        label = "Mask OK" if compliant else "No Mask"
        cv2.putText(display_frame, label, (dx1, dy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        if not compliant:
            violators += 1

    missing = [f"{violators} person(s) without mask"] if violators > 0 else []

    # ── State machine ─────────────────────────────────────────────
    now = time.time()

    if not person_found:
        _reset(state, grace)
        state["status"] = "IDLE"

    elif violators == 0:
        if state["status"] != "COMPLIANT":
            _reset(state, grace)
            insert_event("safety_mask", "COMPLIANT", [], state.get("detected", []))
        state["status"] = "COMPLIANT"

    else:
        if state["violation_start"] is None:
            state["violation_start"] = now
            state["staff_notified"]  = False
            insert_event("safety_mask", "WARNING", missing, state.get("detected", []))

        elapsed = now - state["violation_start"]
        if elapsed < grace:
            state["status"]    = "WARNING"
            state["countdown"] = round(grace - elapsed, 1)
        else:
            state["status"] = "ALERT"
            if not state["staff_notified"]:
                event_id = insert_event("safety_mask", "ALERT", missing, state.get("detected", []))
                state["alert_event_id"] = event_id
                state["staff_notified"] = True
                print(f"[SafetyMask] ALERT event_id={event_id}")

    state["missing"] = missing
    return display_frame, state


def _reset(state, grace):
    state["violation_start"]   = None
    state["staff_notified"]    = False
    state["countdown"]         = grace
    state["last_person_boxes"] = []
    state["person_miss_count"] = 0
    state["recording_started"] = False
    state["alert_event_id"]    = None


# =====================================================================
# MAIN  (new — this is what makes it runnable stand-alone)
# =====================================================================
def draw_status_bar(display_frame, state):
    color_map = {
        "IDLE":      WHITE,
        "COMPLIANT": GREEN,
        "WARNING":   CYAN,
        "ALERT":     RED,
    }
    color = color_map.get(state["status"], WHITE)

    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

    text = f"STATUS: {state['status']}"
    if state["status"] == "WARNING":
        text += f"  (ALERT in {state['countdown']}s)"

    cv2.putText(display_frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_frame, f"Detected classes: {', '.join(state['detected']) or 'none'}",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


def main():
    init(CONFIG)
    state = fresh_state()

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open webcam.")
            return

    window_name = "Safety Mask Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    last_alert_time = 0

    print("Safety mask monitor running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        infer_frame = frame
        display_frame = frame.copy()

        display_frame, state = process_frame(display_frame, infer_frame, state)
        draw_status_bar(display_frame, state)

        # ------------------------------------------------------------
        # Voice Alerts (does not affect state machine / status logic)
        # ------------------------------------------------------------
        current_time = time.time()

        if current_time - last_alert_time > ALERT_COOLDOWN:

            if state["status"] == "WARNING":
                trigger_voice_alert("Warning. Please wear your face mask.")
                last_alert_time = current_time

            elif state["status"] == "ALERT":
                trigger_voice_alert("Critical safety breach. Face mask is missing.")
                last_alert_time = current_time

        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
