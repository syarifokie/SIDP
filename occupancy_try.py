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
        writer.writerow([timestamp, f"Occupancy Change to {count}", status, count])
    print(f"[{timestamp}] LOGGED: {status} ({count} people)")

# =====================================================================
# ALERTING MODULE
# =====================================================================
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)

def trigger_voice_alert(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# =====================================================================
# MAIN SYSTEM
# =====================================================================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

prev_frame_time = 0
last_status = "COMPLIANT"
status_stable_frames = 0
REQUIRED_STABLE_FRAMES = 15
last_alert_time = 0
ALERT_COOLDOWN = 7

# =====================================================================
# THRESHOLD: Warning triggers when people count EXCEEDS this number
# =====================================================================
CAPACITY_LIMIT = 1  # Changed from 10 to 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Performance
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    # 2. Computer Vision
    results = model(frame, verbose=False, conf=0.45)
    current_frame_count = 0

    # 3. Draw Bounding Boxes around each detected person
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # Class 0 = person
            current_frame_count += 1

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # Choose box color based on current status
            if current_frame_count > CAPACITY_LIMIT:
                box_color = (0, 0, 255)   # Red if over limit
            else:
                box_color = (0, 255, 0)   # Green if within limit

            # Draw rectangle around person
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw label above the box
            label = f"Person {current_frame_count} ({confidence:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_bg_y = max(y1 - 20, 0)
            cv2.rectangle(frame,
                          (x1, label_bg_y),
                          (x1 + label_size[0] + 6, label_bg_y + label_size[1] + 6),
                          box_color, -1)
            cv2.putText(frame, label,
                        (x1 + 3, label_bg_y + label_size[1] + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Decision Logic
    if current_frame_count <= CAPACITY_LIMIT:
        detected_status = "COMPLIANT"
        status_color = (0, 255, 0)   # Green
        display_text = f"STATUS: COMPLIANT ({current_frame_count} Pers.)"
    else:
        detected_status = "ALERT"
        status_color = (0, 0, 255)   # Red
        display_text = f"STATUS: ALERT - CAPACITY EXCEEDED! ({current_frame_count} Pers.)"

    # 5. State machine stability check
    if detected_status != last_status:
        status_stable_frames += 1
        if status_stable_frames >= REQUIRED_STABLE_FRAMES:
            last_status = detected_status
            status_stable_frames = 0
            log_event(last_status, current_frame_count)
    else:
        status_stable_frames = 0

    # 6. Voice Alert
    if last_status == "ALERT" and (time.time() - last_alert_time > ALERT_COOLDOWN):
        trigger_voice_alert("Alert. Laboratory occupancy limit exceeded.")
        last_alert_time = time.time()

    # 7. Dashboard HUD
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (20, 20, 20), -1)
    cv2.putText(frame, display_text,
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Limit: {CAPACITY_LIMIT} person(s)",
                (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("UTP Lab Safety Integration Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()