def warp_perspective(img):
    """
    Takes image and changes the perspectipe to crerate a birds eye view image.
    
    Parameters:
    img (numpy array): numpy array form of the image
    
    Returns:
    numpy array: bards eye perspective view of the image in the form of numpy array
    """
    import numpy as np
    import cv2
    
    img_size = img.shape
    
    bottom_left=[190, img_size[0]] #left bottom most point of trapezium
    bottom_right=[1200, img_size[0]] #right bottom most point of trapezium
    top_left=[590, img_size[0]//2 + 90] # left top most point of trapezium
    top_right=[700, img_size[0]//2 + 90] # right top most point of trapezium
    
    offset = 250
    # source points for image warp
    src= np.float32([bottom_left,top_left,bottom_right,top_right]) 
    # destination points for image warp
    dst= np.float32([[offset , img_size[0]], [offset  ,0], 
                     [img_size[1] - offset, img_size[0]], [img_size[1] - offset, 0]]) 
    
    # get the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # change the perspective and return
    return cv2.warpPerspective(img, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
    
def ROI(original_image):
    """
    Takes an image and draws region of interest as a trapezoid on that image; 
    it overwrites on the input image itself
    
    Parameters:
    img (numpy array): numpy array form of the image
    """
    import numpy as np
    import cv2
    
    img_size = original_image.shape
    bottom_left=[190, img_size[0]] #left bottom most point of trapezium
    bottom_right=[1200, img_size[0]] #right bottom most point of trapezium
    top_left=[590, img_size[0]//2 + 90] # left top most point of trapezium
    top_right=[700, img_size[0]//2 + 90] # right top most point of trapezium
    # fit the trapezoid
    cv2.polylines(original_image,np.int32(np.array([[bottom_left,top_left,top_right,bottom_right]])),True,(0,0,255),10)
    return None