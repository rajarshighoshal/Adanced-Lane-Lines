def color_extractor(img, color_space, threshold, color_channel):
    import numpy as np
    import cv2
    
    colorspace = cv2.cvtColor(img, color_space)
    extracted_channel = colorspace[:,:,color_channel]
    binary = np.zeros_like(extracted_channel)
    binary[(extracted_channel >= threshold[0]) & (extracted_channel <= threshold[1])] = 1
    return binary

