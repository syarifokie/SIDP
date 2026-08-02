"""
Plugin: Occupancy Counting
Logic: stable frame counter before state change (prevents flickering)
Exposes: init(config), process_frame(display_frame, infer_frame, state)
"""
import cv2
import time
from database import insert_event

_cfg   = {}
GREEN  = (50,  205, 50)
RED    = (60,  60,  220)
CYAN = (255, 220, 0)   # add at top


def init(config):
    global _cfg

    _cfg = config

    print(f"[Occupancy] Ready — limit: {config.get('max_occupancy', 2)}")


def fresh_state():
    return {
        "status":              "IDLE",
        "violation_start":     None,
        "staff_notified":      False,
        "countdown":           0,
        "missing":             [],
        "detected":            [],
        "count":               0,
        "last_status":         "COMPLIANT",
        "stable_frames":       0,
        "recording_started":   False,
        "alert_event_id":      None,
    }


def process_frame(display_frame, infer_frame, state, person_results=None):
    max_occ       = _cfg.get("max_occupancy", 2)
    grace         = _cfg.get("grace_period", 3.0)
    stable_needed = _cfg.get("stable_frames", 15)  # frames before state change

    sx = display_frame.shape[1] / infer_frame.shape[1]
    sy = display_frame.shape[0] / infer_frame.shape[0]

    if person_results is None:
        return display_frame, state

    results = person_results
    count = len(results[0].boxes)

    # # ── Draw bounding boxes with person numbering ─────────────────
    # for i, box in enumerate(results[0].boxes, start=1):
    #     x1, y1, x2, y2 = box.xyxy[0]
    #     conf_val = float(box.conf[0])
    #     color    = RED if count > max_occ else CYAN
    #     x1s, y1s = int(x1*sx), int(y1*sy)
    #     x2s, y2s = int(x2*sx), int(y2*sy)

    #     cv2.rectangle(display_frame, (x1s, y1s), (x2s, y2s), color, 1)

    #     label     = f"Person {i} ({conf_val:.0%})"
    #     (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #     label_y   = max(y1s - 20, 0)
    #     cv2.rectangle(display_frame,
    #                   (x1s, label_y), (x1s + lw + 6, label_y + lh + 6),
    #                   color, -1)
    #     cv2.putText(display_frame, label,
    #                 (x1s + 4, y1s + 16),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Occupancy count overlay ───────────────────────────────────
    occ_text = f"Occupancy: {count}/{max_occ}"
    color_occ = RED if count > max_occ else GREEN
    cv2.putText(display_frame, occ_text,
                (display_frame.shape[1]//2 - 70, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_occ, 2, cv2.LINE_AA)

    # ── Determine raw status this frame ──────────────────────────
    if count == 0:
        detected_status = "IDLE"
    elif count <= max_occ:
        detected_status = "COMPLIANT"
    else:
        detected_status = "UNSAFE"

    # ── Stable frame counter — prevents flickering ────────────────
    if detected_status != state["last_status"]:
        state["stable_frames"] += 1
        if state["stable_frames"] >= stable_needed:
            state["last_status"]   = detected_status
            state["stable_frames"] = 0
    else:
        state["stable_frames"] = 0

    confirmed_status = state["last_status"]
# ── State machine ─────────────────────────────────────────────
    now = time.time()

    if confirmed_status == "IDLE":
        _reset(state, grace)
        state["status"] = "IDLE"

    elif confirmed_status == "COMPLIANT":
        if state["status"] != "COMPLIANT":
            _reset(state, grace)
            insert_event("occupancy", "COMPLIANT", [], [f"{count} person(s)"])
        state["status"] = "COMPLIANT"

    else:
        # Go straight to ALERT — no WARNING grace period
        state["status"] = "ALERT"
        if not state["staff_notified"]:
            event_id = insert_event("occupancy", "ALERT",
                                    [f"Over limit ({count}/{max_occ})"], [])
            state["alert_event_id"] = event_id
            state["staff_notified"] = True
            print(f"[Occupancy] ALERT event_id={event_id}")

    state["count"]    = count
    state["missing"]  = [f"Over limit: {count}/{max_occ}"] if count > max_occ else []
    state["detected"] = [f"{count} person(s)"]
    return display_frame, state


def _reset(state, grace):
    state["violation_start"]  = None
    state["staff_notified"]   = False
    state["countdown"]        = grace
    state["recording_started"] = False
    state["alert_event_id"]   = None
