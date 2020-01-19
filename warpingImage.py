def warp_perspective(img):
    """
    Takes image and changes the perspectipe to crerate a birds eye view image.
    
    Parameters:
    img (numpy array): numpy array form of the image
    
    Returns:
    numpy array: birds eye perspective view of the image in the form of numpy array
    numpy array: transformation matrix
    numpy array: reverse transformation matrix
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
    
    # get the transform matrix and reveerse transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # change the perspective 
    warp_view = cv2.warpPerspective(img, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
    return warp_view, M, M_inv

    
def ROI(original_image):
    """
    Takes an image and draws region of interest as a trapezoid on that image; 
    it overwrites on the input image itself
    
    Parameters:
    img (numpy array): numpy array form of the image
    
    Returns:
    numpyt array: numpy array form of the masked image containing only region of interest
    """
    import numpy as np
    import cv2
    
    img_size = original_image.shape
    bottom_left=[190, img_size[0]] #left bottom most point of trapezoid
    bottom_right=[1200, img_size[0]] #right bottom most point of trapezoid
    top_left=[590, img_size[0]//2 + 90] # left top most point of trapezoid
    top_right=[700, img_size[0]//2 + 90] # right top most point of trapezoid
    vertices = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
    
    #defining a blank mask to start with
    mask = np.zeros_like(original_image)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img_size) > 2:
        channel_count = img_size[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(original_image, mask)
    return masked_image