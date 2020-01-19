def camera_calibration(img_path_list):
    """
    Takes chessboard image path list as input and produces 
    camera matrix and distortion coefficient as output
    
    Given a list of image paths of 9X6 chessboard images, 
    the function goes through the list and get 3D object points 
    based on 2D image points in chessboard corners.
    Once it found image points and object points it applies camera 
    calibration function to get camera matrix and distortion coefficients
    
    Parameters:
    img_path_list (list): a list containing all the image paths of 9X6 chessboards 
    
    Returns:
    numpy array: numpy array of camera matrix
    numpy array: numpy array containing distortion coefficients
    """
    import numpy as np
    import matplotlib.image as mpimg
    import glob
    import cv2
    
    
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
    """
    Takes distorted image, camera matrix and distortion coefficents as input
    and produces undistorted image as output.
    
    Parameters:
    image (numpy array): numpy array form of the distorted image
    mtx (numpy array): camera matrix
    dist (numpy array): distortion coefficients 
    
    Returns:
    numpy array: undistorted image in the form of numpy array
    """
    
    import cv2
    return cv2.undistort(image, mtx, dist, None, mtx)