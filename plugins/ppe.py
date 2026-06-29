"""
Plugin: PPE Compliance Detection
Exposes: init(config), process_frame(frame, infer_frame, state)
"""
import cv2
import time
from database import insert_event

_cfg          = {}
_ppe_model    = None
_person_model = None

GREEN  = (50,  205, 50)
AMBER  = (0,   165, 255)
RED    = (60,  60,  220)
CYAN   = (255, 220, 0)


def init(config):
    """Load models and store config. Called once at startup."""
    global _cfg, _ppe_model, _person_model
    _cfg = config

    from ultralytics import YOLO
    _ppe_model    = YOLO(config["model_path"])
    _person_model = YOLO(config["person_model_path"])
    print(f"[PPE] Models loaded — classes: {_ppe_model.names}")


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
    """
    Run detection on infer_frame, annotate display_frame, update state.
    Returns (display_frame, state).
    """
    conf      = _cfg.get("conf_threshold", 0.40)
    required  = set(_cfg.get("required_items", []))
    labels    = _cfg.get("labels", {})
    grace     = _cfg.get("grace_period", 3.0)

    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    # --- Person detection ---
    p_results    = _person_model(infer_frame, verbose=False, conf=conf, classes=[0])
    person_found = len(p_results[0].boxes) > 0

    for box in p_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(display_frame,
                      (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
                      CYAN, 2)
        cv2.putText(display_frame, f"Person ({float(box.conf[0]):.2f})",
                    (int(x1*sx), int(y1*sy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

    # --- PPE detection ---
    ppe_results = _ppe_model(infer_frame, verbose=False, conf=conf)
    detected    = set()

    for box in ppe_results[0].boxes:
        cls_name   = _ppe_model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detected.add(cls_name)
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(display_frame,
                      (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
                      GREEN, 2)
        cv2.putText(display_frame, f"{cls_name} ({confidence:.2f})",
                    (int(x1*sx), int(y1*sy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)

    missing_keys   = [k for k in required if k not in detected]
    missing_labels = [labels.get(k, k) for k in missing_keys]

    # --- State machine ---
    now = time.time()

    if not person_found:
        _reset(state, grace)
        state["status"] = "IDLE"

    elif not missing_keys:
        if state["status"] != "COMPLIANT":
            _reset(state, grace)
            insert_event("ppe", "COMPLIANT", [], list(detected))
        state["status"] = "COMPLIANT"

    else:
        if state["violation_start"] is None:
            state["violation_start"] = now
            state["staff_notified"]  = False
            insert_event("ppe", "WARNING", missing_labels, list(detected))

        elapsed = now - state["violation_start"]

        if elapsed < grace:
            state["status"]    = "WARNING"
            state["countdown"] = round(grace - elapsed, 1)
        else:
            state["status"] = "ALERT"
            if not state["staff_notified"]:
                insert_event("ppe", "ALERT", missing_labels, list(detected))
                state["staff_notified"] = True

    state["missing"]  = missing_labels
    state["detected"] = list(detected)
    return display_frame, state


def _reset(state, grace):
    state["violation_start"] = None
    state["staff_notified"]  = False
    state["countdown"]       = grace