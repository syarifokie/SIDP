import cv2
import numpy as np
from ultralytics import YOLO


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_PATH = "model/yolo26n.pt"

SAFE_DISTANCE = 1.0         # metres - radius of "personal space" around each person

ASSUMED_HEIGHT = 1.7         # metres - average person height, used to estimate depth
CAMERA_FOV_HORIZONTAL = 78.0   # degrees - rough horizontal field of view of your webcam
                             # (60 is a common default for laptop/USB webcams; check
                             # your camera's spec sheet for a more accurate value)


# ==============================================================
# LOAD YOLO
# ==============================================================

model = YOLO(MODEL_PATH)


# ==============================================================
# CAMERA
# ==============================================================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# focal length in pixels, derived from field of view (pinhole camera model)
FOCAL_PX = (FRAME_WIDTH / 2) / np.tan(np.radians(CAMERA_FOV_HORIZONTAL / 2))


# ==============================================================
# FULLSCREEN WINDOW
# ==============================================================

cv2.namedWindow("Distance Monitoring", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Distance Monitoring",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)


# ==============================================================
# MAIN LOOP
# ==============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_width = frame.shape[1]

    results = model(frame, verbose=False, conf=0.45)

    people = []
    boxes = []

    for box in results[0].boxes:

        if int(box.cls[0]) == 0:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bbox_height = max(y2 - y1, 1)   # avoid divide by zero
            centre_x = (x1 + x2) / 2

            # estimated depth (distance straight out from camera)
            depth = (ASSUMED_HEIGHT * FOCAL_PX) / bbox_height

            # estimated lateral offset (left/right position)
            lateral = (depth * (centre_x - frame_width / 2)) / FOCAL_PX

            people.append((lateral, depth))
            boxes.append((x1, y1, x2, y2))

    violations = []
    minimum_distance = 999

    for i in range(len(people)):
        for j in range(i + 1, len(people)):

            distance = np.linalg.norm(
                np.array(people[i]) - np.array(people[j])
            )

            minimum_distance = min(minimum_distance, distance)

            if distance < SAFE_DISTANCE:
                violations.append((i, j, distance))

    # ==========================================================
    # DRAW PEOPLE
    # ==========================================================

    for index, box in enumerate(boxes):

        colour = (0, 255, 0)

        for v in violations:
            if index in v[:2]:
                colour = (0, 0, 255)

        x1, y1, x2, y2 = box

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        cv2.putText(
            frame,
            f"Person {index + 1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )

    # draw violation lines

    for i, j, d in violations:

        p1 = boxes[i]
        p2 = boxes[j]

        centre1 = (int((p1[0] + p1[2]) / 2), p1[3])
        centre2 = (int((p2[0] + p2[2]) / 2), p2[3])

        cv2.line(frame, centre1, centre2, (0, 0, 255), 3)

    # ============…