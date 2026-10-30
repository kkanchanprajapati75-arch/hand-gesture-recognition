import cv2
import mediapipe as mp
import time

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

with HandLandmarker.create_from_options(options) as detector:
    start_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame not received!")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = int((time.time() - start_time) * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(
                frame, "Hand Detected!", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
