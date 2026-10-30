import cv2
import mediapipe as mp
import joblib
import time
from collections import deque, Counter

model = joblib.load("gesture_model.pkl")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

options = Options(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

history = deque(maxlen=10)
last_timestamp = 0

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame not received!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = time.monotonic_ns() // 1_000_000
        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(image, timestamp)

        label = "Show Your Hand"
        color = (0, 0, 255)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            features = []

            for point in hand:
                features.extend([point.x, point.y, point.z])

            probabilities = model.predict_proba([features])[0]
            best_index = probabilities.argmax()
            confidence = probabilities[best_index]
            prediction = model.classes_[best_index]

            if confidence >= 0.30:
                history.append(prediction)

                if history:
                    label = Counter(history).most_common(1)[0][0]
                    color = (0, 255, 0)
            else:
                history.clear()
                label = "Uncertain"
                color = (0, 165, 255)

            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        else:
            history.clear()

        cv2.putText(
            frame,
            label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Improved Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("Camera closed. Thank you!")
