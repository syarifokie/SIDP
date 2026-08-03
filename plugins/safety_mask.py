"""
Plugin: Safety Mask Detection
Filters ppe.pt detections to mask class only.
Exposes: init(config), process_frame(display_frame, infer_frame, state)
"""
import cv2
import time
from database import insert_event

_cfg          = {}
_mask_model = None

GREEN = (50,  205, 50)
RED   = (60,  60,  220)
CYAN  = (255, 220, 0)


def init(config):
    global _cfg, _mask_model

    _cfg = config

    from ultralytics import YOLO
    import torch

    if torch.cuda.is_available():
        device = "cuda:0"
        print("[GPU] Mask model using CUDA")
    else:
        device = "cpu"
        print("[GPU] Mask model using CPU")

    _mask_model = YOLO(
        config["model_path"]
    )

    _mask_model.to(device)

    print(f"[SafetyMask] Model loaded on {device}")


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
        "mask_boxes":        [],
        "detection_history": [],   # rolling True/False per frame
    }


def _has_mask_near_person(person_box, mask_boxes, threshold=0.15):
    px1, py1, px2, py2 = person_box
    upper_y2 = py1 + (py2 - py1) * 0.7

    for mx1, my1, mx2, my2 in mask_boxes:
        mask_cx = (mx1 + mx2) / 2
        mask_cy = (my1 + my2) / 2

        # Check if mask centre falls within person box horizontally
        # and within upper 70% vertically
        if px1 <= mask_cx <= px2 and py1 <= mask_cy <= upper_y2:
            return True

        # Fallback — IoU check
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

    # ── Shared Person detection ─────────────────────────────────
    if person_results is not None:

        if len(person_results[0].boxes) > 0:
            state["last_person_boxes"] = person_results[0].boxes
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
            cls_name   = _mask_model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detected.add(cls_name)

            if cls_name == mask_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask_boxes.append((x1, y1, x2, y2))

                # Draw mask bounding box in bright green
                dx1, dy1 = int(x1*sx), int(y1*sy)
                dx2, dy2 = int(x2*sx), int(y2*sy)
                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), GREEN, 2)
                cv2.putText(display_frame,
                            f"Mask ({confidence:.2f})",
                            (dx1, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)

        state["detected"]   = list(detected)
        state["mask_boxes"] = mask_boxes
    else:
        mask_boxes = state.get("mask_boxes", [])

        # Redraw cached mask boxes on skipped frames
        for (x1, y1, x2, y2) in mask_boxes:
            dx1, dy1 = int(x1*sx), int(y1*sy)
            dx2, dy2 = int(x2*sx), int(y2*sy)
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), GREEN, 2)
            cv2.putText(display_frame, "Mask",
                        (dx1, dy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)

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

    # ── Majority vote — false positive filter ────────────────────
    history_window   = _cfg.get("history_window", 10)    # frames to look back
    violation_ratio  = _cfg.get("violation_ratio", 0.7)  # 70% must be violations

    # Append this frame's result (True = compliant, False = violation)
    state["detection_history"].append(violators == 0)

    # Keep only the last N frames
    if len(state["detection_history"]) > history_window:
        state["detection_history"].pop(0)

    # Need full window before making a decision
    if len(state["detection_history"]) < history_window:
        state["status"]  = "IDLE"   # accumulating — hold neutral
        state["missing"] = []
        return display_frame, state

    # Calculate ratio of compliant frames in window
    compliant_ratio    = sum(state["detection_history"]) / len(state["detection_history"])
    confirmed_violation = compliant_ratio < (1 - violation_ratio)
    # confirmed_violation = True means 70%+ frames showed violation

    # ── State machine — driven by majority vote ───────────────────
    now = time.time()

    if not person_found:
        _reset(state, grace)
        state["status"] = "IDLE"

    elif not confirmed_violation:
        # Majority says compliant — reset
        if state["status"] != "COMPLIANT":
            _reset(state, grace)
            insert_event("safety_mask", "COMPLIANT", [], state.get("detected", []))
        state["status"] = "COMPLIANT"

    else:
        # Majority confirmed violation — proceed to WARNING/UNSAFE
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
