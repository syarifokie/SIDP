import cv2
import time
import os
import csv
from datetime import datetime
import pyttsx3
from ultralytics import YOLO

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
MODEL_PATH = os.path.join("model", "yolo26n.pt")
LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "lab_safety_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event Type", "Safety Level", "Person Count"])


def log_event(status, count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            f"Occupancy Change to {count}",
            status,
            count
        ])

    print(f"[{timestamp}] LOGGED: {status} ({count} people)")


# =====================================================================
# ALERTING MODULE
# =====================================================================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)


def trigger_voice_alert(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


# =====================================================================
# LOAD YOLO MODEL
# =====================================================================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# =====================================================================
# SYSTEM VARIABLES
# =====================================================================
prev_frame_time = 0

last_status = "COMPLIANT"
status_stable_frames = 0
REQUIRED_STABLE_FRAMES = 15

last_alert_time = 0
ALERT_COOLDOWN = 7

# =====================================================================
# OCCUPANCY RULE
# Compliant : <= 2 People
# Unsafe    : > 2 People
# =====================================================================
CAPACITY_LIMIT = 2


# =====================================================================
# MAIN LOOP
# =====================================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -------------------------------------------------------------
    # FPS Calculation
    # -------------------------------------------------------------
    new_frame_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0

    prev_frame_time = new_frame_time

    # -------------------------------------------------------------
    # Run YOLO Detection
    # -------------------------------------------------------------
    results = model(frame, verbose=False, conf=0.45)

    person_boxes = []

    for box in results[0].boxes:

        if int(box.cls[0]) == 0:
            person_boxes.append(box)

    current_frame_count = len(person_boxes)

    # -------------------------------------------------------------
    # Determine Current Status
    # -------------------------------------------------------------
    if current_frame_count <= CAPACITY_LIMIT:

        detected_status = "COMPLIANT"
        status_color = (0, 255, 0)

        display_text = (
            f"STATUS : COMPLIANT "
            f"({current_frame_count}/{CAPACITY_LIMIT} People)"
        )

        is_alert = False

    else:

        detected_status = "UNSAFE"
        status_color = (0, 0, 255)

        display_text = (
            f"STATUS : UNSAFE - OCCUPANCY LIMIT EXCEEDED "
            f"({current_frame_count} People)"
        )

        is_alert = True

    # -------------------------------------------------------------
    # Draw Bounding Boxes
    # -------------------------------------------------------------
    for i, box in enumerate(person_boxes, start=1):

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        box_color = (0, 0, 255) if is_alert else (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label = f"Person {i} ({confidence:.0%})"

        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        label_bg_y = max(y1 - 20, 0)

        cv2.rectangle(
            frame,
            (x1, label_bg_y),
            (x1 + label_size[0] + 6, label_bg_y + label_size[1] + 6),
            box_color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 3, label_bg_y + label_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # -------------------------------------------------------------
    # State Machine (Prevents Flickering)
    # -------------------------------------------------------------
    if detected_status != last_status:

        status_stable_frames += 1

        if status_stable_frames >= REQUIRED_STABLE_FRAMES:

            last_status = detected_status

            status_stable_frames = 0

            log_event(last_status, current_frame_count)

    else:

        status_stable_frames = 0

    # -------------------------------------------------------------
    # Voice Alert
    # -------------------------------------------------------------
    if last_status == "UNSAFE":

        if time.time() - last_alert_time > ALERT_COOLDOWN:

            trigger_voice_alert(
                "Warning. More than two people detected inside the laboratory."
            )

            last_alert_time = time.time()

    # -------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 100),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "LAB OCCUPANCY MONITORING SYSTEM",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        display_text,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"People Detected : {current_frame_count}",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )

    cv2.putText(
        frame,
        f"Maximum Allowed : {CAPACITY_LIMIT}",
        (270, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )

    cv2.putText(
        frame,
        f"FPS : {fps:.1f}",
        (520, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )

    # -------------------------------------------------------------
    # Display Window
    # -------------------------------------------------------------
    cv2.imshow("UTP Lab Occupancy Monitoring Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =====================================================================
# CLEANUP
# =====================================================================
cap.release()
cv2.destroyAllWindows()