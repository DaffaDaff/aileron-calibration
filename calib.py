import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Create vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Define real world coordinates for points in the checkerboard pattern
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('calib.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If desired number of corners are found, refine the pixel coordinates and display
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Perform camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel coordinates of the detected corners (imgpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera calibration results
np.savez("calibration_data.npz", mtx=mtx, dist=dist)
