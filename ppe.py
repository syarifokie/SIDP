
import cv2
import time
import os
import csv
from datetime import datetime
import pyttsx3
import threading
from ultralytics import YOLO

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
MODEL_PATH = os.path.join("model", "ppe.pt")
LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "ppe_safety_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event Status", "Detected PPE"])


def log_event(status, items):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, status, items])

    print(f"[{timestamp}] LOGGED PPE EVENT: {status} | {items}")
    


# =====================================================================
# BACKGROUND VOICE ENGINE
# =====================================================================
def _speak_worker(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def trigger_voice_alert(text):
    threading.Thread(
        target=_speak_worker,
        args=(text,),
        daemon=True
    ).start()


# =====================================================================
# LOAD MODEL
# =====================================================================
print(f"Loading PPE AI Model : {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("Model Classes:")
print(model.names)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open webcam.")
        exit()


WINDOW_NAME = "UTP Lab Safety Integration Dashboard - PPE Engine"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1024, 768)

prev_frame_time = 0

current_status = "SAFE"

violation_start_time = None

last_alert_time = 0
ALERT_COOLDOWN = 6

print("\n========================================")
print("LAB PPE COMPLIANCE MONITOR ACTIVE")
print("Monitoring:")
print(" - Face Mask")
print(" - Hand Gloves")
print("========================================")
print("Press Q to quit.\n")

# =====================================================================
# MAIN LOOP
# =====================================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------
    new_frame_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0

    prev_frame_time = new_frame_time

    # ------------------------------------------------------------
    # YOLO Detection
    # ------------------------------------------------------------
    results = model(frame, verbose=False, conf=0.40)

    detected_items = set()

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        detected_items.add(class_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label = f"{class_name} ({confidence:.2f})"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2
        )

    # ------------------------------------------------------------
    # REQUIRED PPE
    # ------------------------------------------------------------
    missing_gear = []

    if "mask" not in detected_items:
        missing_gear.append("Face Mask")

    if "gloves" not in detected_items:
        missing_gear.append("Hand Gloves")

    # ------------------------------------------------------------
    # PPE State Machine
    # ------------------------------------------------------------
    if len(missing_gear) == 0:

        if current_status != "SAFE":

            current_status = "SAFE"

            log_event(
                current_status,
                ", ".join(sorted(detected_items))
            )

        violation_start_time = None

        status_color = (0, 255, 0)

        display_text = "PPE STATUS : COMPLIANT"

    else:

        current_time = time.time()

        if violation_start_time is None:

            violation_start_time = current_time

            current_status = "WARNING"

            log_event(
                current_status,
                ", ".join(sorted(detected_items))
            )

        elapsed = current_time - violation_start_time

        remaining = max(0, 5.0 - elapsed)

        if elapsed >= 5:

            if current_status != "UNSAFE":

                current_status = "UNSAFE"

                log_event(
                    current_status,
                    ", ".join(sorted(detected_items))
                )

            status_color = (0, 0, 255)

            display_text = (
                "CRITICAL : MISSING "
                + ", ".join(missing_gear)
            )

        else:

            current_status = "WARNING"

            status_color = (0, 255, 255)

            display_text = (
                f"WARNING : Equip {', '.join(missing_gear)} "
                f"({remaining:.1f}s)"
            )

    # ------------------------------------------------------------
    # Voice Alerts
    # ------------------------------------------------------------
    current_time = time.time()

    if current_time - last_alert_time > ALERT_COOLDOWN:

        if current_status == "WARNING":

            trigger_voice_alert(
                "Warning. Please wear your face mask and hand gloves."
            )

            last_alert_time = current_time

        elif current_status == "UNSAFE":

            trigger_voice_alert(
                "Critical safety breach. Face mask or hand gloves are missing."
            )

            last_alert_time = current_time

    # ------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], 90),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    cv2.putText(
        frame,
        display_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"PPE Required : Face Mask + Hand Gloves",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (230, 230, 230),
        1
    )

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (frame.shape[1] - 120, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (230, 230, 230),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================================
# CLEANUP
# =====================================================================
cap.release()
cv2.destroyAllWindows()