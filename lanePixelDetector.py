# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # initialize all others not carried over between first detections
        self.reset()

    def reset(self):
        # flag of line detection
        self.detected = False  
        # recent polynomial coefficients
        self.recent_fit = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.x = None  
        # y values for detected line pixels
        self.y = None
        # counter to reset after 5 iterations if issues arise
        self.counter = 0        

    def count_check(self, n):
        ''' 
        Resets the line class upon failing five times in a row.
        '''
        # Increment the counter
        self.counter += 1
        # Reset if failed n times
        if self.counter >= n:
            self.reset()

    def fit_line(self, x_points, y_points, first_try=True):
        '''
        Fit a second order polynomial to the line.
        The challenge videos sometimes throws errors, so the below trys first.
        Upon the error being thrown, either reset the line or add to counter.
        '''
        n = 5
        self.current_fit = np.polyfit(y_points, x_points, 2)
        self.all_x = x_points
        self.all_y = y_points
        self.recent_fit.append(self.current_fit)
        if len(self.recent_fit) > 1:
            self.diffs = (self.recent_fit[-2] - self.recent_fit[-1]) / self.recent_fit[-2]
        self.recent_fit = self.recent_fit[-n:]
        self.best_fit = np.mean(self.recent_fit, axis = 0)
        line_fit = self.current_fit
        self.detected = True
        self.counter = 0
        return line_fit

#         except (TypeError, np.linalg.LinAlgError):
#             line_fit = self.best_fit
#             if first_try == True:
#                 self.reset()
#             else:
#                 self.count_check()

#             return line_fit


def find_first_lanes(bin_image, ym_per_pix, xm_per_pix, nwindows=25, margin=100, minpix=50):
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