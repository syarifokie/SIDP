"""
alarm.py — Priority audio alert system for SIDP
Single speaker thread reads from a priority queue.
No overlapping speech, no duplicate messages.
"""
import threading
import queue
import time
import pyttsx3

# =====================================================================
# PRIORITY LEVELS (lower number = higher priority)
# =====================================================================
PRIORITY = {
    "ALERT": 1,
    "WARNING": 2,
    "IDLE":  3,
}

# =====================================================================
# PER-USE-CASE ALERT MESSAGES
# =====================================================================
ALERT_MESSAGES = {
    "ppe":       "PPE violation detected. Ensure all protective equipment is worn.",
    "occupancy": "Occupancy limit exceeded. Please reduce the number of people in the lab.",
    "no_phone":  "Phone use detected. No phones allowed in this zone.",
}

WARNING_MESSAGES = {
    "ppe":       "Warning. Missing protective equipment. Please comply immediately.",
    "occupancy": "Warning. Approaching occupancy limit.",
    "no_phone":  "Warning. Phone detected. Please put it away.",
}

# Spoken when ALL active use cases are in ALERT simultaneously
COMBINED_WARNING = "Warning. Multiple safety violations detected. Please address all issues immediately."
COMBINED_ALERT = "Critical safety breach. Multiple violations detected. Please take immediate action."


# =====================================================================
# INTERNAL STATE
# =====================================================================
_speech_queue  = queue.PriorityQueue()  # (priority, sequence, text)
_speaker_thread = None
_running        = False
_lock           = threading.Lock()
_seq            = 0                     # tiebreaker to preserve insertion order
# Track the last spoken message for cooldown
_last_spoken = {}


# =====================================================================
# PUBLIC API
# =====================================================================
def start():
    """Start the speaker thread. Call once at startup."""
    global _speaker_thread, _running
    with _lock:
        if _running:
            return
        _running = True
        _speaker_thread = threading.Thread(target=_speaker_loop, daemon=True)
        _speaker_thread.start()
        print("[Alarm] Speaker thread started")

def stop():
    """Stop the speaker thread gracefully."""
    global _running
    _running = False
    # Unblock the queue.get() call
    _speech_queue.put((99, 0, None))

def notify(use_case, status, active_alerts, active_warnings=None, combined_cooldown=10.0):
    global _seq

    if active_warnings is None:
        active_warnings = set()

    if status not in ("WARNING", "ALERT"):
        _last_spoken.pop(use_case, None)
        return

    now      = time.time()

    # Combined WARNING
    if status == "WARNING" and len(active_warnings) >= 2:
        combined_key  = "_combined_warning"
        last_combined = _last_spoken.get(combined_key)
        if not last_combined or (now - last_combined[1]) >= combined_cooldown:
            _enqueue(PRIORITY["WARNING"], COMBINED_WARNING)
            _last_spoken[combined_key] = ("WARNING", now)
        return

    # Combined ALERT
    if status == "ALERT" and len(active_alerts) >= 2:
        combined_key  = "_combined_alert"
        last_combined = _last_spoken.get(combined_key)
        if not last_combined or (now - last_combined[1]) >= combined_cooldown:
            _enqueue(PRIORITY["ALERT"], COMBINED_ALERT)
            _last_spoken[combined_key] = ("ALERT", now)
        return

    msg = ALERT_MESSAGES.get(use_case) if status == "ALERT" else WARNING_MESSAGES.get(use_case)
    if msg:
        _enqueue(PRIORITY[status], msg)


# def notify(use_case, status, active_alerts):
#     """
#     Called by main.py process() on each frame.
 
#     use_case     — 'ppe', 'occupancy', 'no_phone'
#     status       — 'WARNING', 'ALERT', 'COMPLIANT', 'IDLE'
#     active_alerts — set of use case names currently in ALERT state
#     """
#     global _seq
 
#     if status not in ("WARNING", "ALERT"):
#         # Clear the cooldown when violation resolves
#         _last_spoken.pop(use_case, None)
#         return
 
#     now      = time.time()
#     last     = _last_spoken.get(use_case)
#     cooldown = _REPEAT_COOLDOWN
 
#     # Check cooldown — skip if same status was recently spoken
#     if last and last[0] == status and (now - last[1]) < cooldown:
#         return
 
#     # If multiple use cases are ALL in ALERT, speak combined message once
#     if status == "ALERT" and len(active_alerts) >= 2:
#         # Use a shared key so combined message isn't repeated per-UC
#         combined_key = "_combined"
#         last_combined = _last_spoken.get(combined_key)
#         if not last_combined or (now - last_combined[1]) >= cooldown:
#             _enqueue(PRIORITY["ALERT"], COMBINED_ALERT)
#             _last_spoken[combined_key] = ("ALERT", now)
#         return
 
#     # Single use case — speak specific message
#     if status == "ALERT":
#         msg = ALERT_MESSAGES.get(use_case)
#     else:
#         msg = WARNING_MESSAGES.get(use_case)
 
#     if msg:
#         _enqueue(PRIORITY[status], msg)
#         _last_spoken[use_case] = (status, now)

def silence():
    """Clear the queue — stops current speech cycle."""
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
        except queue.Empty:
            break
    print("[Alarm] Silenced")

def is_active():
    """Returns True if there are pending messages."""
    return not _speech_queue.empty()

# =====================================================================
# INTERNAL
# =====================================================================
def _enqueue(priority, text):
    global _seq
    _seq += 1
    _speech_queue.put((priority, _seq, text))

def _speaker_loop():
    """
    Single thread — reads from priority queue and speaks one at a time.
    Never overlaps. Highest priority spoken first.
    """
    print("[Alarm] Speaker ready")
    while _running:
        try:
            priority, seq, text = _speech_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if text is None:   # stop signal
            break

        _speak(text)

def _speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Alarm] Speech error: {e}")
