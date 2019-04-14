import cv2
import os
from os import path
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
from time import sleep
import pyautogui
import shutil
import subprocess
import glob
import collections
import math


global line_lt, line_rt

ym_per_pix = 30 / 720   # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

time_window = 10        # results are averaged over this number of frames
test_img_dir = 'frames'
calib_images_dir='camera_cal'




class Line:
    def __init__(self, buffer_len=10):

        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.

        :param new_fit_pixel: new polynomial coefficients (pixel)
        :param new_fit_meter: new polynomial coefficients (meter)
        :param detected: if the Line was detected or inferred
        :param clear_buffer: if True, reset state
        :return: None
        """
        self.detected = detected

        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # radius of curvature of the line (averaged)
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])





def process_pipeline(frame, keep_state=True):
    line_lt = Line(buffer_len=time_window)  # line on the left of the lane
    line_rt = Line(buffer_len=time_window)  # line on the right of the lane

    assert path.exists(calib_images_dir), '"{}" must exist and contain calibration images.'.format(calib_images_dir)# checks for the existance of cal images

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    for filename in images:

        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if pattern_found is True:         
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # undistort the image using coefficients found in calibration
    img_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    # binarize the frame s.t. lane lines are highlighted as much as possible

    yellow_HSV_th_min = np.array([0, 70, 70])
    yellow_HSV_th_max = np.array([50, 255, 255])
    
    h, w = img_undistorted.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by threshold in HSV color space
    HSV = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2HSV)

    min_th_ok = np.all(HSV > yellow_HSV_th_min, axis=2)
    max_th_ok = np.all(HSV < yellow_HSV_th_max, axis=2)

    HSV_yellow_mask = np.logical_and(min_th_ok, max_th_ok)
    binary = np.logical_or(binary, HSV_yellow_mask)

    # highlight white lines by thresholding the equalized frame

    gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, eq_white_mask = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    
    binary = np.logical_or(binary, eq_white_mask)
    
    kernel_size=9

    gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

    sobel_mask = sobel_mag.astype(bool)

    binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)

    img_binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # compute perspective transform to obtain bird's eye view
    h, w = img_binary.shape[:2]

    src = np.float32([[w, h-10], [0, h-10], [546, 460], [732, 460]])  # tr
    dst = np.float32([[w, h], [0, h], [0, 0], [w, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    img_birdeye = cv2.warpPerspective(img_binary, M, (w, h), flags=cv2.INTER_LINEAR)


    # fit 2-degree polynomial curve onto lane lines found
    if keep_state and line_lt.detected and line_rt.detected:
            height, width = birdeye_binary.shape

            left_fit_pixel = line_lt.last_fit_pixel
            right_fit_pixel = line_rt.last_fit_pixel

            nonzero = birdeye_binary.nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            margin = 100
            left_lane_inds = (
            (nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & (
            nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
            right_lane_inds = (
            (nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & (
            nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))

            # Extract left and right line pixel positions
            line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
            line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

            detected = True
            if not list(line_lt.all_x) or not list(line_lt.all_y):
                left_fit_pixel = line_lt.last_fit_pixel
                left_fit_meter = line_lt.last_fit_meter
                detected = False
            else:
                left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
                left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

            if not list(line_rt.all_x) or not list(line_rt.all_y):
                right_fit_pixel = line_rt.last_fit_pixel
                right_fit_meter = line_rt.last_fit_meter
                detected = False
            else:
                right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
                right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

            line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
            line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

            # Generate x and y values for plotting
            ploty = np.linspace(0, height - 1, height)
            left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
            right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

            # Create an image to draw on and an image to show the selection window
            img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
            window_img = np.zeros_like(img_fit)

            # Color in left and right line pixels
            img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)
    else:

            height, width = img_birdeye.shape
            n_windows=9

            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(img_birdeye[height//2:-30, :], axis=0)

            # Create an output image to draw on and  visualize the result
            img_fit = np.dstack((img_birdeye, img_birdeye, img_birdeye)) * 255

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = len(histogram) // 2
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Set height of windows
            window_height = np.int(height / n_windows)

            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = img_birdeye.nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            margin = 100  # width of the windows +/- margin
            minpix = 50   # minimum number of pixels found to recenter window

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(n_windows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = height - (window + 1) * window_height
                win_y_high = height - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Draw the windows on the visualization image
                cv2.rectangle(img_fit, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(img_fit, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                                  & (nonzero_x < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                                   & (nonzero_x < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
            line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

            detected = True
            if not list(line_lt.all_x) or not list(line_lt.all_y):
                left_fit_pixel = line_lt.last_fit_pixel
                left_fit_meter = line_lt.last_fit_meter
                detected = False
            else:
                left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
                left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

            if not list(line_rt.all_x) or not list(line_rt.all_y):
                right_fit_pixel = line_rt.last_fit_pixel
                right_fit_meter = line_rt.last_fit_meter
                detected = False
            else:
                right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
                right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)
    
            line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
            line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

            # Generate x and y values for plotting
            ploty = np.linspace(0, height - 1, height)
            left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
            right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

            img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]


    # compute offset in meter from center of the lane
    global offset_meter
    frame_width=frame.shape[1]
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = float((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter1 = xm_per_pix * offset_pix
        offset_meter = float('{:.2f}'.format(offset_meter1))
    else:
        offset_meter = "Error"


    # draw the surface enclosed by lane lines back onto the original frame

    height, width, _ = img_undistorted.shape

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

    blend_onto_road = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
    line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    blend_on_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)    

    # stitch on the top of final output images from different steps of the pipeline

    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_output = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_output[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_output[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_output[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    global mean_curvature_meter
    mean_curvature_meter = int(np.mean([line_lt.curvature_meter, line_rt.curvature_meter]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_output, 'Curvature radius: {:.0f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_output, 'Offset from center: {:.2f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_output


def findroad(anglein):
    countdoc = open('numframes.txt', 'r')
    count = int(countdoc.readline())
    countdoc.close()
    countdoc = open('numframes.txt', 'w')
    newcount = count + 1
    countdoc.write("%.0f" % newcount)
    countdoc.close()    
    test_img = 'img%.0f.jpg' % count
    makeframe = pyautogui.screenshot('frames\img%.0f.jpg'% count)
    frame = cv2.imread(os.path.join(test_img_dir, test_img))
    blend = process_pipeline(frame)
    cv2.imwrite('output_images/{}'.format(test_img), blend)
    
    #fix horizonal offset of camera caused by bike lean
    
    radius = 1 #height from ground to camera when bike is upright in meters

    angle = math.radians(anglein)

    sint = math.sin(angle)

    cost = math.cos(angle)

    h = radius*sint #horizontal offset left of right of the camera as the bike tilts

    if abs(h) < .5: #5 is # of cm of horizontal offset allowed before digital correction sets in
        pass
    else:
        offset_meter = offset_meter + h

    return mean_curvature_meter, offset_meter

    
def init():
    countdoc = open('numframes.txt', 'w')
    countdoc.write('0')
    countdoc.close()
    dest1 = 'output_images_overflow/'
    dest2 = 'frames_overflow/'
    files = os.listdir(test_img_dir)
    
    for f in files:
            shutil.move('frames/' + f, dest1)

    files = os.listdir('output_images')

    for f in files:
            shutil.move('output_images/' + f, dest2)

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    subprocess.Popen(["CamDesk.exe"] + ["-d", "CamDesk.exe"], startupinfo=startupinfo)
    sleep(5)#waits 3 seconds for CamDesk to open
    pyautogui.hotkey('fn', 'f11')
    pyautogui.moveTo(1366, 768)#moves mouse out of the visible computer screen (on a screeen with 1366 by 768 resolution) 
