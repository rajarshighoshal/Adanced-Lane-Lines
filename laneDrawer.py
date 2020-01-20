def draw_patch(img, left_fit, right_fit, M):
    """
    Draw the lane lines on the image.
    
    Parameters:
    img (numpy array): numpy array form of the original image
    left_fit (numpy array): indices of best fit points of left lane
    right_fit (numpy array): indices of best fit points of right lane
    M (numpy array): reverse transformation matrix of warping operation
    
    Returns:
    numpy array: output image
    """
    import numpy as np
    import cv2
    
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Calculate points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255,255,0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M, (img.shape[1], img.shape[0])) 
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def calculate_curvature(y_range, left_fit_cr, ym_per_pix):
    """
    Returns the curvature of the polynomial fit on the y range 
    
    Parameters:
    y_range (float): range of y in the image
    left_fit_cr (numpy array): left lane curve coefficients after polynomial fitting
    ym_per_pix (float): meter length per pixel for y direction
    
    Returns:
    float : curvature value of the lane curve
    """
    import numpy as np
    return ((1 + (2*left_fit_cr[0]*y_range*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])