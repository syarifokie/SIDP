"""
Standalone Proximity / Social Distance Monitor
Self-contained version of the original plugin (init/process_frame/fresh_state)
so it can be run directly with: python proximity_standalone.py

Replaces the external `database.insert_event` call with a simple CSV logger,
and replaces the external main-loop/config injection with a built-in webcam
loop and CONFIG dict.
"""
import cv2
import time
import os
import csv
from datetime import datetime
import numpy as np

# =====================================================================
# CONFIG  (previously passed in via init(config) from the host app)
# =====================================================================
CONFIG = {
    "model_path": "yolov8n.pt",        # any YOLO model with a "person" class (class 0)
    "conf_threshold": 0.45,
    "safe_distance": 1.0,              # metres
    "assumed_height": 1.7,             # metres, average person height
    "camera_fov_horizontal": 78.0,     # degrees, adjust to your webcam's FOV
    "grace_period": 3.0,               # seconds before WARNING escalates to ALERT
}

LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "proximity_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Type", "Status", "Missing", "Detected"])

GREEN  = (50,  205, 50)
RED    = (60,  60,  220)
YELLOW = (0,   200, 255)
WHITE  = (255, 255, 255)

_model = None


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
    global _model
    from ultralytics import YOLO
    _model = YOLO(config["model_path"])
    print(f"[Proximity] Model loaded — safe distance: {config.get('safe_distance', 1.0)}m")


def fresh_state():
    return {
        "status":            "IDLE",
        "violation_start":   None,
        "staff_notified":    False,
        "countdown":         CONFIG.get("grace_period", 3.0),
        "missing":           [],
        "detected":          [],
        "violations":        [],
        "recording_started": False,
        "alert_event_id":    None,
    }


def process_frame(display_frame, infer_frame, state, person_results=None, run_person=True, run_ppe=True):
    conf           = CONFIG.get("conf_threshold", 0.45)
    safe_distance  = CONFIG.get("safe_distance", 1.0)
    assumed_height = CONFIG.get("assumed_height", 1.7)
    fov_h          = CONFIG.get("camera_fov_horizontal", 78.0)
    grace          = CONFIG.get("grace_period", 3.0)

    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    frame_w   = infer_frame.shape[1]
    focal_px  = (frame_w / 2) / np.tan(np.radians(fov_h / 2))

    results = _model(infer_frame, verbose=False, conf=conf, classes=[0])

    people = []
    boxes  = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_h   = max(y2 - y1, 1)
        centre_x = (x1 + x2) / 2
        depth    = (assumed_height * focal_px) / bbox_h
        lateral  = (depth * (centre_x - frame_w / 2)) / focal_px
        people.append((lateral, depth))
        boxes.append((x1, y1, x2, y2))

    violations = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            dist = float(np.linalg.norm(np.array(people[i]) - np.array(people[j])))
            if dist < safe_distance:
                violations.append((i, j, dist))

    violating_indices = {idx for v in violations for idx in v[:2]}
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = RED if idx in violating_indices else WHITE
        x1s, y1s = int(x1 * sx), int(y1 * sy)
        x2s, y2s = int(x2 * sx), int(y2 * sy)
        cv2.rectangle(display_frame, (x1s, y1s), (x2s, y2s), color, 1)
        cv2.putText(display_frame, f"Person {idx+1}", (x1s, y1s + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    for i, j, dist in violations:
        b1, b2  = boxes[i], boxes[j]
        centre1 = (int((b1[0] + b1[2]) / 2 * sx), int(b1[3] * sy))
        centre2 = (int((b2[0] + b2[2]) / 2 * sx), int(b2[3] * sy))
        cv2.line(display_frame, centre1, centre2, RED, 2)
        mid = ((centre1[0] + centre2[0]) // 2, (centre1[1] + centre2[1]) // 2)
        cv2.putText(display_frame, f"{dist:.2f}m", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)

    now     = time.time()
    missing = [f"{len(violations)} proximity violation(s)"] if violations else []

    if len(people) == 0:
        _reset(state, grace)
        state["status"] = "IDLE"

    elif not violations:
        if state["status"] != "COMPLIANT":
            _reset(state, grace)
            insert_event("proximity", "COMPLIANT", [], [f"{len(people)} person(s)"])
        state["status"] = "COMPLIANT"

    else:
        if state["violation_start"] is None:
            state["violation_start"] = now
            state["staff_notified"]  = False
            insert_event("proximity", "WARNING", missing, [f"{len(people)} person(s)"])

        elapsed = now - state["violation_start"]
        if elapsed < grace:
            state["status"]    = "WARNING"
            state["countdown"] = round(grace - elapsed, 1)
        else:
            state["status"] = "ALERT"
            if not state["staff_notified"]:
                event_id = insert_event("proximity", "ALERT", missing, [f"{len(people)} person(s)"])
                state["alert_event_id"] = event_id
                state["staff_notified"] = True
                print(f"[Proximity] ALERT event_id={event_id}")

    state["missing"]    = missing
    state["detected"]   = [f"{len(people)} person(s)"]
    state["violations"] = violations
    return display_frame, state


def _reset(state, grace):
    state["violation_start"]   = None
    state["staff_notified"]    = False
    state["countdown"]         = grace
    state["recording_started"] = False
    state["alert_event_id"]    = None


# =====================================================================
# MAIN  (new — this is what makes it runnable stand-alone)
# =====================================================================
def draw_status_bar(display_frame, state):
    color_map = {
        "IDLE":      WHITE,
        "COMPLIANT": GREEN,
        "WARNING":   YELLOW,
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
    cv2.putText(display_frame, f"Detected: {', '.join(state['detected']) or 'none'}",
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

    window_name = "Proximity / Social Distance Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    print("Proximity monitor running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # infer_frame and display_frame can be the same resolution here;
        # kept separate to preserve the original scaling logic (sx/sy).
        infer_frame = frame
        display_frame = frame.copy()

        display_frame, state = process_frame(display_frame, infer_frame, state)
        draw_status_bar(display_frame, state)

        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()