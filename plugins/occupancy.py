"""
Plugin: Occupancy Counting
Exposes: init(config), process_frame(frame, infer_frame, state)
"""
import cv2
import time
from database import insert_event

_cfg   = {}
_model = None
CYAN   = (255, 220, 0)
RED    = (60,  60,  220)
GREEN  = (50,  205, 50)


def init(config):
    global _cfg, _model
    _cfg = config
    from ultralytics import YOLO
    _model = YOLO(config["model_path"])
    print(f"[Occupancy] Model loaded")


def fresh_state():
    return {
        "status":          "IDLE",
        "violation_start": None,
        "staff_notified":  False,
        "countdown":       _cfg.get("grace_period", 3.0),
        "missing":         [],
        "detected":        [],
        "count":           0,
    }


def process_frame(display_frame, infer_frame, state):
    conf      = _cfg.get("conf_threshold", 0.40)
    max_occ   = _cfg.get("max_occupancy", 5)
    grace     = _cfg.get("grace_period", 3.0)

    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    results = _model(infer_frame, verbose=False, conf=conf, classes=[0])
    count   = len(results[0].boxes)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        color = RED if count > max_occ else GREEN
        cv2.rectangle(display_frame,
                      (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
                      color, 2)

    cv2.putText(display_frame, f"Occupancy: {count}/{max_occ}",
                (display_frame.shape[1]//2 - 60, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                RED if count > max_occ else GREEN, 2, cv2.LINE_AA)

    now = time.time()

    if count == 0:
        _reset(state, grace)
        state["status"] = "IDLE"

    elif count <= max_occ:
        if state["status"] != "COMPLIANT":
            _reset(state, grace)
            insert_event("occupancy", "COMPLIANT", [], [f"{count} person(s)"])
        state["status"] = "COMPLIANT"

    else:
        if state["violation_start"] is None:
            state["violation_start"] = now
            state["staff_notified"]  = False
            insert_event("occupancy", "WARNING",
                         [f"Over limit ({count}/{max_occ})"], [])

        elapsed = now - state["violation_start"]

        if elapsed < grace:
            state["status"]    = "WARNING"
            state["countdown"] = round(grace - elapsed, 1)
        else:
            state["status"] = "ALERT"
            if not state["staff_notified"]:
                insert_event("occupancy", "ALERT",
                             [f"Over limit ({count}/{max_occ})"], [])
                state["staff_notified"] = True

    state["count"]    = count
    state["missing"]  = [f"Over limit: {count}/{max_occ}"] if count > max_occ else []
    state["detected"] = [f"{count} person(s)"]
    return display_frame, state


def _reset(state, grace):
    state["violation_start"] = None
    state["staff_notified"]  = False
    state["countdown"]       = grace