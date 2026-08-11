"""
Plugin: Proximity / Social Distance Monitoring
Uses pinhole camera model to estimate real-world distance between people.
Exposes: init(config), process_frame(display_frame, infer_frame, state)
"""
import cv2
import time
import numpy as np
from database import insert_event

_cfg   = {}
GREEN  = (50,  205, 50)
RED    = (60,  60,  220)
YELLOW = (0,   200, 255)
WHITE = (255, 255, 255)   # add at top

def init(config):
    global _cfg

    _cfg = config

    print(f"[Proximity] Ready — safe distance: {config.get('safe_distance', 1.0)}m")


def fresh_state():
    return {
        "status":            "IDLE",
        "violation_start":   None,
        "staff_notified":    False,
        "countdown":         _cfg.get("grace_period", 3.0),
        "missing":           [],
        "detected":          [],
        "violations":        [],
        "recording_started": False,
        "alert_event_id":    None,
    }


def process_frame(display_frame, infer_frame, state, person_results=None):
    safe_distance  = _cfg.get("safe_distance", 1.0)       # metres
    assumed_height = _cfg.get("assumed_height", 1.7)      # metres
    fov_h          = _cfg.get("camera_fov_horizontal", 78.0)  # degrees
    grace          = _cfg.get("grace_period", 3.0)

    # Scale factors infer → display
    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    frame_w   = infer_frame.shape[1]
    focal_px  = (frame_w / 2) / np.tan(np.radians(fov_h / 2))

    if person_results is None:
        return display_frame, state

    results = person_results

    people = []   # (lateral, depth)
    boxes  = []   # (x1, y1, x2, y2) in infer coords

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_h   = max(y2 - y1, 1)
        centre_x = (x1 + x2) / 2
        depth    = (assumed_height * focal_px) / bbox_h
        lateral  = (depth * (centre_x - frame_w / 2)) / focal_px
        people.append((lateral, depth))
        boxes.append((x1, y1, x2, y2))

    # ── Calculate pairwise distances ──────────────────────────────
    violations = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            dist = float(np.linalg.norm(
                np.array(people[i]) - np.array(people[j])
            ))
            if dist < safe_distance:
                violations.append((i, j, dist))

    # # # ── Draw bounding boxes ───────────────────────────────────────
    # violating_indices = {idx for v in violations for idx in v[:2]}

    # for idx, box in enumerate(boxes):
    #     x1, y1, x2, y2 = box
    #     color = RED if idx in violating_indices else WHITE
    #     x1s, y1s = int(x1*sx), int(y1*sy)
    #     x2s, y2s = int(x2*sx), int(y2*sy)
    #     cv2.rectangle(display_frame, (x1s, y1s), (x2s, y2s), color, 1)
    #     cv2.putText(display_frame, f"Person {idx+1}",
    #                 (x1s, y1s  + 32),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # ── Draw violation lines ──────────────────────────────────────
    for i, j, dist in violations:
        b1, b2   = boxes[i], boxes[j]
        centre1  = (int((b1[0]+b1[2])/2 * sx), int(b1[3] * sy))
        centre2  = (int((b2[0]+b2[2])/2 * sx), int(b2[3] * sy))
        cv2.line(display_frame, centre1, centre2, RED, 2)
        mid      = ((centre1[0]+centre2[0])//2, (centre1[1]+centre2[1])//2)
        cv2.putText(display_frame, f"{dist:.2f}m",
                    mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)

    # ── State machine ─────────────────────────────────────────────
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
                event_id = insert_event("proximity", "ALERT", missing,
                                        [f"{len(people)} person(s)"])
                state["alert_event_id"] = event_id
                state["staff_notified"] = True
                print(f"[Proximity] ALERT event_id={event_id}")

    state["missing"]     = missing
    state["detected"]    = [f"{len(people)} person(s)"]
    state["violations"]  = violations
    return display_frame, state


def _reset(state, grace):
    state["violation_start"]  = None
    state["staff_notified"]   = False
    state["countdown"]        = grace
    state["recording_started"] = False
    state["alert_event_id"]   = None
