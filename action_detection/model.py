import mediapipe as mp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# Function to extract body keypoints using Mediapipe
mp_pose = mp.solutions.pose
def extract_body_keypoints(frame):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            return None
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y))
        return keypoints

def extract_features(keypoints):
    left_shoulder_index = 11
    left_elbow_index = 13
    left_hip_index = 23
    left_knee_index = 25

    right_shoulder_index = 12
    right_elbow_index = 14
    right_hip_index = 24
    right_knee_index = 26

    # Extract the coordinates of the relevant keypoints
    left_shoulder = keypoints[left_shoulder_index]
    left_elbow = keypoints[left_elbow_index]
    left_hip = keypoints[left_hip_index]
    left_knee = keypoints[left_knee_index]

    right_shoulder = keypoints[right_shoulder_index]
    right_elbow = keypoints[right_elbow_index]
    right_hip = keypoints[right_hip_index]
    right_knee = keypoints[right_knee_index]

    # Compute the angles using vector operations
    left_shoulder_elbow = np.array(left_elbow) - np.array(left_shoulder)
    left_hip_knee = np.array(left_knee) - np.array(left_hip)

    right_shoulder_elbow = np.array(right_elbow) - np.array(right_shoulder)
    right_hip_knee = np.array(right_knee) - np.array(right_hip)

    left_shoulder_angle = np.arctan2(left_shoulder_elbow[1], left_shoulder_elbow[0])
    left_hip_angle = np.arctan2(left_hip_knee[1], left_hip_knee[0])

    right_shoulder_angle = np.arctan2(right_shoulder_elbow[1], right_shoulder_elbow[0])
    right_hip_angle = np.arctan2(right_hip_knee[1], right_hip_knee[0])

    # Convert angles from radians to degrees
    left_shoulder_angle_deg = np.degrees(left_shoulder_angle)
    left_hip_angle_deg = np.degrees(left_hip_angle)

    right_shoulder_angle_deg = np.degrees(right_shoulder_angle)
    right_hip_angle_deg = np.degrees(right_hip_angle)

    # Combine the extracted features into a single array
    features = np.array([left_shoulder_angle_deg, left_hip_angle_deg, right_shoulder_angle_deg, right_hip_angle_deg])
    return features

# Directory paths for train and test data
train_dir = "train"
test_dir = "test"


actions = ["running", "jumping"]  

X = []
y = []

for action_idx, action in enumerate(actions):
    action_path = os.path.join(train_dir, action)
    for video_file in os.listdir(action_path):
        video_path = os.path.join(action_path, video_file)
        video_capture = cv2.VideoCapture(video_path)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            keypoints = extract_body_keypoints(frame)
            if keypoints is not None:
                features = extract_features(keypoints)
                X.append(features)
                y.append(action_idx)
        
        video_capture.release()

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)  
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load and preprocess the test video
test_video_path = "test/test.mp4"
video_capture = cv2.VideoCapture(test_video_path)

# Lists to store the extracted features and predicted actions for each frame
test_features = []
predicted_actions = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    keypoints = extract_body_keypoints(frame)
    if keypoints is not None:
        features = extract_features(keypoints)
        test_features.append(features)

# Convert the list of features to a numpy array
test_features = np.array(test_features)

# Make predictions using the trained model
predictions = knn_classifier.predict(test_features)

# Map the predicted labels to action names
action_names = ["running", "jumping"]  # Add more actions if needed
predicted_actions = [action_names[prediction] for prediction in predictions]

# Print the predicted actions for each frame
for i, action in enumerate(predicted_actions):
    print(f"Frame {i + 1}: {action}")

# Release the video capture
video_capture.release()