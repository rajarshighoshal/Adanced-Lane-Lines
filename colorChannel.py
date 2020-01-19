def create_threshold_binary(undist_img, s_thresh=(125, 255), 
                            sx_thresh=(10, 100), r_thresh = (200, 255), sobel_kernel = 3):
    """
    Takes an image and based on thresholds provided creates an binary masked image of the input image
    
    Parameters:
    undist_img (numpy array): numpy array form of the image
    s_thresh (tuple): threshold for S color space
    sx_thresh (tuple): threshold for grdient in the x direction of the sobel of L color space
    r_thresh (tuple): threshold for R color space
    sobel_kernel (int): kernal to apply for sobel operation (must be an odd number)
    
    Returns:
    numpyt array: numpy array form of the finaal binary image
    """
    import cv2
    import numpy as np
    
    # R channel
    r_channel = undist_img[:,:,0]
    # Convert to HLS colorspace
    hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # sobelx - takes the derivate in x, absolute value, then rescale
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) 
             & (scaled_sobelx <= sx_thresh[1])] = 1

    # threshold R color channel
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
    # threshold S color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # if two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (sxbinary == 1)) 
                     | ((sxbinary == 1) & (r_binary == 1))
                     | ((s_binary == 1) & (r_binary == 1))] = 1

    return combined_binary