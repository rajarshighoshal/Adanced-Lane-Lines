def find_lanes(bin_image, ym_per_pix, xm_per_pix, nwindows=25, margin=100, minpix=50):
    """
    Takes the binary image and find left and right lane lines without any previous findings.

    Parameters:
    bin_image (numpy array): binary masked image after warping
    ym_per_pix (float): meter length per pixel for y direction
    xm_per_pix (float): meter length per pixel for x direction
    nwindows (int): number of sliding windows 
    margin (int): margin in which area search for best pixel values
    minpix (int): minimum of number of pixels below which x and y position don't chnage

    Returns:
    numpy array: left lane curve coefficients after polynomial fitting
    numpy array: right lane curve coefficients after polynomial fitting
    numpy array: left lane curve coefficients after polynomial fitting in meters
    numpy array: right lane curve coefficients after polynomial fitting in meters
    numpy array: indices where right lane pixels are found 
    numpy array: indices where left lane pixels are found
    numpy array: output image after drawing sliding windows
    numpy array: all the nonzero x values
    numpy array: all the nonzero y values
    """
    import numpy as np
    import cv2

    # create a histogram
    hist = np.sum(bin_image[bin_image.shape[0]//2:,:], axis=0) 
    # output image
    out =  np.dstack((bin_image, bin_image, bin_image))*255
    # find left peak
    leftx_base = np.argmax(hist[:hist.shape[0]//2])
    # find right peak
    rightx_base = np.argmax(hist[hist.shape[0]//2:]) + hist.shape[0]//2
    # height of window
    window_height = bin_image.shape[0]//nwindows
    # find non-zero pixels
    nonzero = bin_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current position of left and right lanes
    left_x_current = leftx_base
    right_x_current = rightx_base
    # create list for left and right points
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = bin_image.shape[0] - (window+1)*window_height
        win_y_high = bin_image.shape[0] - window*window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin
        # draw the windows on the visualization image
        cv2.rectangle(out,(win_x_left_low,win_y_low),(win_x_left_high,win_y_high),(0,255,0),2) 
        cv2.rectangle(out,(win_x_right_low,win_y_low),(win_x_right_high,win_y_high),(0,255,0),2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & 
                          (nonzeroy < win_y_high) & 
                          (nonzerox >= win_x_left_low) & 
                          (nonzerox < win_x_left_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & 
                           (nonzeroy < win_y_high) & 
                           (nonzerox >= win_x_right_low) & 
                           (nonzerox < win_x_right_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            right_x_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    return left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out, nonzerox, nonzeroy