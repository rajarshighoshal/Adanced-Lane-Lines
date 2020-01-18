import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def camera_calibration(img_path_list):
    # create object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # arrays to store objpionts and imgpoints
    objpoints = [] # array to store 3D real object points
    imgpoints = [] # array to store 2D image points

    # loop through all the images
    for img_name in img_path_list:
        # read image
        img = mpimg.imread(img_name)
        # convert image to gray
        grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(grayImg, (9,6), None)
        # if corners foud append to imgpoints
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    
    # calibrates with camera and retuen calibratation coeffiecients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1], None, None)
    
    return mtx, dist


def undistort_img(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)