import cv2
import time
import os
import csv
from datetime import datetime
import pyttsx3
import threading
from ultralytics import YOLO

# =====================================================================
# OPTION 1: CONFIDENCE THRESHOLD FIX
# Change: conf raised from 0.35 → 0.60
# =====================================================================

MODEL_PATH = os.path.join("model", "yolo26n.pt")
LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "phone_restriction_log_option1.csv")

os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event Status", "Details"])

def log_event(status, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, status, details])
    print(f"[{timestamp}] LOGGED PHONE EVENT: {status} | {details}")

def _speak_worker(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

def trigger_voice_alert(text):
    threading.Thread(target=_speak_worker, args=(text,), daemon=True).start()

print(f"Loading Model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        exit()

WINDOW_NAME = "OPTION 1 — Confidence Threshold (conf=0.60)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1024, 768)

prev_frame_time = 0
current_status = "SAFE"
violation_start_time = None
last_alert_time = 0
ALERT_COOLDOWN = 6

# =====================================================================
# OPTION 1 SETTING — Only change from original
# =====================================================================
CONFIDENCE_THRESHOLD = 0.60  # Was 0.35 — higher = stricter = fewer false positives

print("\n=== OPTION 1: HIGH CONFIDENCE THRESHOLD MODE ===")
print(f"Confidence threshold set to: {CONFIDENCE_THRESHOLD}")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    # OPTION 1: Higher confidence threshold applied here
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    phone_detected = False
    phones_count = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "cell phone":
            phone_detected = True
            phones_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"PHONE ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if not phone_detected:
        if current_status != "SAFE":
            current_status = "SAFE"
            log_event(current_status, "No restricted devices on screen.")
        violation_start_time = None
        status_color = (0, 255, 0)
        display_text = "ZONE STATUS: SECURE (NO PHONES)"
    else:
        current_time = time.time()
        if violation_start_time is None:
            violation_start_time = current_time
            current_status = "WARNING"
            log_event(current_status, "Phone detected on screen.")

        elapsed_violation_time = current_time - violation_start_time
        countdown_left = max(0, 5.0 - elapsed_violation_time)

        if elapsed_violation_time >= 5.0:
            if current_status != "UNSAFE":
                current_status = "UNSAFE"
                log_event(current_status, f"CRITICAL: Continuous phone breach. Devices: {phones_count}")
            status_color = (0, 0, 255)
            display_text = f"CRITICAL BREACH: PROHIBITED DEVICE ACTIVE FOR {int(elapsed_violation_time)}s!"
        else:
            current_status = "WARNING"
            status_color = (0, 255, 255)
            display_text = f"WARNING: Put phone away in {countdown_left:.1f}s!"

    current_time = time.time()
    if current_time - last_alert_time > ALERT_COOLDOWN:
        if current_status == "WARNING":
            trigger_voice_alert("Warning. Mobile device detected in restricted laboratory zone.")
            last_alert_time = current_time
        elif current_status == "UNSAFE":
            trigger_voice_alert("Critical violation. Put your phone away immediately.")
            last_alert_time = current_time

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 95), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, display_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"[OPTION 1] Confidence Threshold: {CONFIDENCE_THRESHOLD}", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 130, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()