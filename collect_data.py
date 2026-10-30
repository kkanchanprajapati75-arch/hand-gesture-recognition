import cv2
import mediapipe as mp
import csv
import os
import time

GESTURES = [
    "Open_Palm", "Fist", "Thumbs_Up", "Thumbs_Down",
    "Victory", "Three_Fingers", "Four_Fingers",
    "OK_Sign", "Rock_Sign", "Call_Me"
]

for gesture in GESTURES:
    os.makedirs(f"dataset/{gesture}", exist_ok=True)

model = "hand_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

options = Options(
    base_options=BaseOptions(model_asset_path=model),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

print("Keys 1-9 and 0 select gestures.")
print("Press S to save a sample. Press Q to quit.")

selected = None
counts = {g: 0 for g in GESTURES}
last_time = 0

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))

        if result.hand_landmarks:
            points = result.hand_landmarks[0]
            for p in points:
                x, y = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        label = selected if selected else "Select gesture (1-0)"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "S=Save | Q=Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if ord('1') <= key <= ord('9'):
            selected = GESTURES[key - ord('1')]
        elif key == ord('0'):
            selected = GESTURES[9]
        elif key == ord('s') and selected and result.hand_landmarks:
            if time.time() - last_time > 0.15:
                filename = f"dataset/{selected}/landmarks.csv"
                values = []
                for p in result.hand_landmarks[0]:
                    values.extend([p.x, p.y, p.z])
                with open(filename, "a", newline="") as f:
                    csv.writer(f).writerow(values)
                counts[selected] += 1
                last_time = time.time()
                print(selected, ":", counts[selected], "samples saved")
        elif key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
print("Data collection finished!")
