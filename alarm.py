"""
alarm.py — Priority audio alert system for SIDP

Features:
- Single speaker thread
- Priority-based speech queue
- No overlapping speech
- Combined alert when multiple violations exist
- Automatically switches back to single alert when one violation is resolved
- Clears everything only when all violations are resolved
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
}


# =====================================================================
# ALERT MESSAGES
# =====================================================================

ALERT_MESSAGES = {
    "safety_mask":
        "Safety mask violation. Ensure mask is worn at all times.",
    "occupancy":
        "Occupancy limit exceeded. Please reduce the number of people in the lab.",
    "proximity":
        "Proximity violation. Please maintain safe distance from others.",
}


WARNING_MESSAGES = {
    "safety_mask":
        "Warning. Safety mask not detected. Please put on your mask.",
    "occupancy":
        "Warning. Approaching occupancy limit.",
    "proximity":
        "Warning. People are too close together. Please maintain distance.",
}


COMBINED_WARNING = (
    "Warning. Multiple safety violations detected. "
    "Please address all issues immediately."
)


COMBINED_ALERT = (
    "Critical safety breach. Multiple violations detected. "
    "Please take immediate action."
)


# =====================================================================
# INTERNAL STATE
# =====================================================================

_speech_queue = queue.PriorityQueue()

_speaker_thread = None
_running = False

_lock = threading.Lock()

_seq = 0

_last_spoken = {}



# =====================================================================
# START / STOP
# =====================================================================

def start():

    global _speaker_thread, _running

    with _lock:

        if _running:
            return

        _running = True

        _speaker_thread = threading.Thread(
            target=_speaker_loop,
            daemon=True
        )

        _speaker_thread.start()

        print("[Alarm] Speaker thread started")



def stop():

    global _running

    _running = False

    _speech_queue.put(
        (99, 0, None)
    )



# =====================================================================
# MAIN NOTIFY FUNCTION
# =====================================================================

def notify(
        use_case,
        status,
        active_alerts,
        active_warnings=None,
        combined_cooldown=10.0):


    global _seq


    if active_warnings is None:
        active_warnings = set()



    now = time.time()



    # ==============================================================
    # COMPLIANT / IDLE
    # ==============================================================

    if status not in ("WARNING", "ALERT"):


        _last_spoken.pop(use_case, None)



        #
        # One violation remains
        #
        if len(active_alerts) == 1:


            _last_spoken.pop(
                "_combined_alert",
                None
            )


            _drain_queue()


            remaining = next(iter(active_alerts))


            msg = ALERT_MESSAGES.get(
                remaining
            )


            if msg:

                _enqueue(
                    PRIORITY["ALERT"],
                    msg
                )


                _last_spoken[remaining] = (
                    "ALERT",
                    now
                )



        #
        # No violation remains
        #
        elif len(active_alerts) == 0:


            _drain_queue()

            _last_spoken.clear()

            print(
                "[Alarm] System compliant"
            )



        return




    # ==============================================================
    # MULTIPLE WARNING
    # ==============================================================

    if status == "WARNING":


        if len(active_warnings) >= 2:


            key = "_combined_warning"


            last = _last_spoken.get(key)



            if (
                last is None
                or now - last[1] >= combined_cooldown
            ):

                _enqueue(
                    PRIORITY["WARNING"],
                    COMBINED_WARNING
                )


                _last_spoken[key] = (
                    "WARNING",
                    now
                )


            return



    # ==============================================================
    # MULTIPLE ALERT
    # ==============================================================


    if status == "ALERT":


        if len(active_alerts) >= 2:


            key = "_combined_alert"


            last = _last_spoken.get(key)



            if (
                last is None
                or now - last[1] >= combined_cooldown
            ):


                _enqueue(
                    PRIORITY["ALERT"],
                    COMBINED_ALERT
                )


                _last_spoken[key] = (
                    "ALERT",
                    now
                )


            return



    # ==============================================================
    # SINGLE VIOLATION
    # ==============================================================


    if status == "ALERT":


        msg = ALERT_MESSAGES.get(
            use_case
        )


    else:


        msg = WARNING_MESSAGES.get(
            use_case
        )



    if msg:


        _enqueue(
            PRIORITY[status],
            msg
        )


        _last_spoken[use_case] = (
            status,
            now
        )



# =====================================================================
# QUEUE CONTROL
# =====================================================================

def silence():

    _drain_queue()

    print(
        "[Alarm] Silenced"
    )



def clear_cooldowns():

    _last_spoken.clear()

    print(
        "[Alarm] Cooldowns cleared"
    )



def is_active():

    return not _speech_queue.empty()



def _enqueue(priority, text):

    global _seq

    _seq += 1

    _speech_queue.put(
        (
            priority,
            _seq,
            text
        )
    )



def _drain_queue():

    while not _speech_queue.empty():

        try:

            _speech_queue.get_nowait()

        except queue.Empty:

            break



# =====================================================================
# SPEAKER THREAD
# =====================================================================

def _speaker_loop():

    print(
        "[Alarm] Speaker ready"
    )


    while _running:


        try:

            priority, seq, text = (
                _speech_queue.get(
                    timeout=0.5
                )
            )


        except queue.Empty:

            continue



        if text is None:

            break



        _speak(text)



def _speak(text):

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            165
        )

        engine.setProperty(
            "volume",
            1.0
        )


        engine.say(text)

        engine.runAndWait()



    except Exception as e:

        print(
            f"[Alarm] Speech error: {e}"
        )
