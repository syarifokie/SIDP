"""
Plugin: No-Phone Zone Monitoring
Exposes: init(config), process_frame(frame, infer_frame, state)
"""
import cv2
import time
from database import insert_event

_cfg   = {}
_model = None
RED    = (60,  60,  220)
GREEN  = (50,  205, 50)


def init(config):
    global _cfg, _model
    _cfg = config
    from ultralytics import YOLO
    _model = YOLO(config["model_path"])
    print(f"[NoPhone] Model loaded")


def fresh_state():
    return {
        "status":          "IDLE",
        "violation_start": None,
        "staff_notified":  False,
        "countdown":       _cfg.get("grace_period", 3.0),
        "missing":         [],
        "detected":        [],
    }


def process_frame(display_frame, infer_frame, state):
    conf         = _cfg.get("conf_threshold", 0.40)
    phone_names  = set(_cfg.get("phone_classes", ["cell phone"]))
    grace        = _cfg.get("grace_period", 3.0)

    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    results     = _model(infer_frame, verbose=False, conf=conf)
    phone_found = False

    for box in results[0].boxes:
        cls_name = _model.names[int(box.cls[0])]
        if cls_name in phone_names:
            phone_found = True
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(display_frame,
                          (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
                          RED, 2)
            cv2.putText(display_frame, f"PHONE DETECTED ({float(box.conf[0]):.2f})",
                        (int(x1*sx), int(y1*sy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)

    now = time.time()

    if not phone_found:
        if state["status"] not in ("IDLE", "COMPLIANT"):
            insert_event("no_phone", "COMPLIANT", [], [])
        _reset(state, grace)
        state["status"] = "COMPLIANT"

    else:
        if state["violation_start"] is None:
            state["violation_start"] = now
            state["staff_notified"]  = False
            insert_event("no_phone", "WARNING", ["Phone in use"], [])

        elapsed = now - state["violation_start"]

        if elapsed < grace:
            state["status"]    = "WARNING"
            state["countdown"] = round(grace - elapsed, 1)
        else:
            state["status"] = "ALERT"
            if not state["staff_notified"]:
                insert_event("no_phone", "ALERT", ["Phone in use"], [])
                state["staff_notified"] = True

    state["missing"]  = ["Phone detected"] if phone_found else []
    state["detected"] = ["cell phone"] if phone_found else []
    return display_frame, state


def _reset(state, grace):
    state["violation_start"] = None
    state["staff_notified"]  = False
    state["countdown"]       = grace
