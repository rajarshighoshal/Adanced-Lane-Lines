class videoPipeline():
    """
    Class for combining all the operations where input will ba a video 
    and output will be another video after applying all the operations
    """
    def __init__(self, input_video, output_video):
        import glob
        from moviepy.editor import VideoFileClip
        from undistortImg import camera_calibration
        # pixel to meter
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # calibrate camera
        img_path_list = glob.glob('/home/workspace/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')
        mtx, dist = camera_calibration(img_path_list)
        self.mtx = mtx
        self.dist = dist
        # convert video to clips
        video_clip = VideoFileClip(input_video)
        # crate left and right lanes
        self.left_fit = None
        self.right_fit = None
        self.left_fit_m = None
        self.right_fit_m = None
        self.left_curvature = None
        self.right_curvature = None
        # apply image processing to evry frames
        clip = video_clip.fl_image(self.process_image)
        # save the generated video
        clip.write_videofile(output_video, audio=False)

    def process_image(self, img):
        """
        Takes image and processes that through all the steps and 
        outputs final image after applying all the operations
        """
        from undistortImg import undistort_img
        from colorChannel import create_threshold_binary
        from warpingImage import warp_perspective
        from lanePixelDetector import find_lanes
        from laneDrawer import draw_patch, calculate_curvature
        import cv2
        import numpy
        
        undist_img = undistort_img(img, self.mtx, self.dist)
        bin_img = create_threshold_binary(undist_img)
        warped_img, M, M_inv = warp_perspective(bin_img)
        try :
            left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = find_lanes(warped_img, self.ym_per_pix, self.xm_per_pix)
        except:
            left_fit, right_fit, left_fit_m, right_fit_m = self.left_fit, self.right_fit, self.left_fit_m, self.right_fit_m
        y_range = img.shape[0]
        left_curvature = calculate_curvature(y_range, left_fit, self.ym_per_pix)
        right_curvature = calculate_curvature(y_range, right_fit, self.ym_per_pix)
        if left_curvature > 10000:
            left_fit = self.left_fit
            left_fit_m = self.left_fit_m
            left_curvature = self.left_curvature
        else:
            self.left_fit = left_fit
            self.left_fit_m = left_fit_m
            self.left_curvature = left_curvature

        if right_curvature > 10000:
            right_fit = self.right_fit
            right_fit_m = self.right_fit_m
            right_curvature = self.right_curvature
        else:
            self.right_fit = right_fit
            self.right_fit_m = right_fit_m
            self.right_curvature = right_curvature
        
        # calculate vehicle center
        x_max = img.shape[1]*self.xm_per_pix
        y_max = img.shape[0]*self.ym_per_pix
        vehicle_center = x_max / 2
        line_left = left_fit_m[0]*y_max**2 + left_fit_m[1]*y_max + left_fit_m[2]
        line_right = right_fit_m[0]*y_max**2 + right_fit_m[1]*y_max + right_fit_m[2]
        line_middle = line_left + (line_right - line_left)/2
        diff_from_vehicle = line_middle - vehicle_center
        if diff_from_vehicle > 0:
            message = '{:.2f} m right'.format(diff_from_vehicle)
        else:
            message = '{:.2f} m left'.format(-diff_from_vehicle)
        output = draw_patch(img, left_fit, right_fit, M_inv)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        cv2.putText(output, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50), font, 2, font_color, 2)
        cv2.putText(output, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 120), font, 2, font_color, 2)
        cv2.putText(output, 'Vehicle is {} of center'.format(message), (50, 190), font, 2, font_color, 2)
        return output