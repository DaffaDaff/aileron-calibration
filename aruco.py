from dronekit import connect, VehicleMode
import time
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

# Connect to the vehicle (replace with your connection string)
vehicle = connect('/dev/ttyACM0', wait_ready=True, baud=57600)  # Update with your connection

def is_parallel(rotation_matrix1, rotation_matrix2, tolerance=5.0):
    """Check if two rotation matrices correspond to parallel orientations."""
    q1 = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix1).as_quat()
    q2 = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix2).as_quat()

    q_relative = scipy.spatial.transform.Rotation.from_quat(q2) * scipy.spatial.transform.Rotation.from_quat(q1).inv()

    relative_euler = q_relative.as_euler('xyz', degrees=True)
    return all(abs(angle) < tolerance for angle in relative_euler), relative_euler

def adjust_servo_trim(relative_euler, aileron_channel=1):
    """Adjust servo trim based on the relative orientation."""
    roll_diff, pitch_diff, yaw_diff = relative_euler

    trim_adjustment = int(roll_diff * 10)  # This is an example scaling factor

    current_trim = vehicle.channels.overrides.get(aileron_channel, 1500)  # Default to midpoint (1500)
    new_trim = current_trim + trim_adjustment
    new_trim = max(1000, min(2000, new_trim))  # Constrain to valid PWM range

    print(f"Adjusting Aileron Trim to: {new_trim}")
    vehicle.channels.overrides[aileron_channel] = new_trim

# Frequency of detection (in seconds)
detection_interval = 0
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the live feed
    cv2.imshow('Aileron Parallel Check', frame)

    # Check if it's time to run the detection logic
    current_time = time.time()
    if current_time - last_detection_time > detection_interval:
        last_detection_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) == 2:  # Ensure exactly two markers are detected
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)

            R1, _ = cv2.Rodrigues(rvecs[0])
            R2, _ = cv2.Rodrigues(rvecs[1])

            parallel, relative_euler = is_parallel(R1, R2)
            
            if parallel:
                cv2.putText(frame, "Aileron is parallel to Wing", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Aileron is NOT parallel to Wing", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                adjust_servo_trim(relative_euler)  # Adjust the aileron trim

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close the vehicle connection
vehicle.close()
