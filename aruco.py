import cv2
import numpy as np
import scipy.spatial.transform

# Load camera calibration data
with np.load('calibration_data.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

marker_length = 0.05  # The actual size of the ArUco marker in meters

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) == 2:  # Ensure exactly two markers are detected
        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

        # Draw the detected markers and their axes
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)

        # Calculate the relative translation vector (tvec)
        relative_tvec = tvecs[1][0] - tvecs[0][0]

        # Calculate the Euclidean distance between the two markers
        relative_distance = np.linalg.norm(relative_tvec)

        # Convert rotation vectors to quaternions
        R1, _ = cv2.Rodrigues(rvecs[0])
        R2, _ = cv2.Rodrigues(rvecs[1])
        
        q1 = scipy.spatial.transform.Rotation.from_matrix(R1).as_quat()
        q2 = scipy.spatial.transform.Rotation.from_matrix(R2).as_quat()

        # Calculate the relative quaternion (rotation from marker 1 to marker 2)
        q_relative = scipy.spatial.transform.Rotation.from_quat(q2) * scipy.spatial.transform.Rotation.from_quat(q1).inv()

        # Convert relative quaternion to Euler angles (roll, pitch, yaw)
        relative_euler = q_relative.as_euler('xyz', degrees=True)
        roll, pitch, yaw = relative_euler

        # Display the relative distance and rotation (roll, pitch, yaw) on the frame
        cv2.putText(frame, f"Distance: {relative_distance:.2f}m", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f} degrees", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('ArUco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
