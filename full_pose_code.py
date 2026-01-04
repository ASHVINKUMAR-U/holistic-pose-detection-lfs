import cv2
import mediapipe as mp
import json
import time

print("HOLISTIC POSE + HANDS + FACE JSON RECORDING STARTED")

# =========================
# MediaPipe Holistic Setup
# =========================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Webcam Setup
# =========================
cap = cv2.VideoCapture(0)
time.sleep(2)

all_frames = []
frame_id = 0

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_id += 1
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)

    # ✅ CORRECT JSON SCHEMA
    frame_data = {
        "frame_id": frame_id,
        "pose": [],
        "left_hand": [],
        "right_hand": [],
        "face": []
    }

    # =========================
    # BODY POSE (33 landmarks)
    # =========================
    if result.pose_landmarks:
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            frame_data["pose"].append({
                "id": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    # =========================
    # LEFT HAND (21 landmarks)
    # =========================
    if result.left_hand_landmarks:
        for idx, lm in enumerate(result.left_hand_landmarks.landmark):
            frame_data["left_hand"].append({
                "id": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })

        mp_draw.draw_landmarks(
            frame,
            result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # =========================
    # RIGHT HAND (21 landmarks)
    # =========================
    if result.right_hand_landmarks:
        for idx, lm in enumerate(result.right_hand_landmarks.landmark):
            frame_data["right_hand"].append({
                "id": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })

        mp_draw.draw_landmarks(
            frame,
            result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # =========================
    # FACE MESH (468 landmarks)
    # =========================
    if result.face_landmarks:
        for idx, lm in enumerate(result.face_landmarks.landmark):
            frame_data["face"].append({
                "id": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })

        mp_draw.draw_landmarks(
            frame,
            result.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(
                thickness=1
            )
        )

    # =========================
    # SAVE FRAME DATA
    # =========================
    all_frames.append(frame_data)

    cv2.imshow("Holistic Pose + Hands + Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup & Save JSON
# =========================
cap.release()
cv2.destroyAllWindows()

print("Total frames recorded:", len(all_frames))

with open("holistic_pose_hands_face.json", "w") as f:
    json.dump(all_frames, f, indent=4)

print("JSON saved as holistic_pose_hands_face.json")
