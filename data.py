import  cv2
import mediapipe as mp
import numpy as np
import os
import time

ACTIONS = np.array(['wave', 'fist', 'grab'])
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
DATA_PATH = os.path.join('MP_Data_V2')
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(detection_results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     detection_results .pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   detection_results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   detection_results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])



cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            sequence_data = []

            for frame_num in range(SEQUENCE_LENGTH):
                ret,frame = cap.read()
                image, results = frame, holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                if frame_num == 0:
                    cv2.putText(image, f'READY? Action: {action}', (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Collector', image)
                    cv2.waitKey(1000)

                cv2.putText(image, f'RECORDING {action} - Seq: {sequence} Frame: {frame_num}', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                if not results.left_hand_landmarks and not results.right_hand_landmarks:
                    cv2.putText(image,"NO HAND DETECTED!",(15,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


                cv2.imshow('Collector', image)

                sequence_data.append(extract_keypoints(results))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            res_path = os.path.join(DATA_PATH, action,f"{action}_{sequence}.npy")
            np.save(res_path, sequence_data)
            print(f"Saved: {res_path}")

    cap.release()
    cv2.destroyAllWindows()




