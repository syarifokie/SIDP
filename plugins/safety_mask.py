"""
Plugin: Safety Mask Detection
Filters ppe.pt detections to mask class only.
Exposes: init(config), process_frame(display_frame, infer_frame, state)
"""
import cv2
import time
from database import insert_event

_cfg          = {}
_mask_model   = None
_person_model = None

GREEN = (50,  205, 50)
RED   = (60,  60,  220)
CYAN  = (255, 220, 0)


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


# For each detected person, check if a mask box overlaps their upper-body region
def _has_mask_near_person(person_box, mask_boxes, threshold=0.3):
    """
    Check if any mask box overlaps the upper half of the person box.
    Upper half = head/face region where mask would appear.
    """
    px1, py1, px2, py2 = person_box
    # Upper half of person box (head region)
    upper_y2 = py1 + (py2 - py1) * 0.5
    upper_box = (px1, py1, px2, upper_y2)

    for mx1, my1, mx2, my2 in mask_boxes:
        # Calculate overlap
        ix1 = max(px1, mx1); iy1 = max(py1, my1)
        ix2 = min(px2, mx2); iy2 = min(upper_y2, my2)
        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2-ix1) * (iy2-iy1)
            mask_area    = (mx2-mx1) * (my2-my1)
            if mask_area > 0 and (intersection/mask_area) >= threshold:
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
        p_results = _person_model(infer_frame, verbose=False,
                                   conf=person_conf, classes=[0])
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

        # Collect all mask boxes first
        for box in ppe_results[0].boxes:
            cls_name = _mask_model.names[int(box.cls[0])]
            detected.add(cls_name)
            if cls_name == mask_class:
                mask_boxes.append(tuple(map(int, box.xyxy[0])))

        state["detected"]  = list(detected)
        state["mask_boxes"] = mask_boxes  # cache for skipped frames
    else:
        mask_boxes = state.get("mask_boxes", [])

    # Check each person individually against cached mask boxes
    for box in state["last_person_boxes"]:
        px1, py1, px2, py2 = map(int, box.xyxy[0])
        dx1, dy1 = int(px1*sx), int(py1*sy)
        dx2, dy2 = int(px2*sx), int(py2*sy)

        compliant = _has_mask_near_person((px1, py1, px2, py2), mask_boxes)
        color     = GREEN if compliant else RED

        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
        label = f"Mask OK" if compliant else "No Mask"
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
                event_id = insert_event("safety_mask", "ALERT", missing,
                                        state.get("detected", []))
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