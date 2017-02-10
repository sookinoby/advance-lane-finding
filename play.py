import numpy as np
import cv2
import pickle

cap = cv2.VideoCapture('project_video.mp4')
#loads the prespective transform matrix
pickle_data = pickle.load(open("transform.p", "rb"))
M = pickle_data["M"]
#calculates the inverse of prespective transform matrix
MInv = np.linalg.inv(M)
pickle_data = pickle.load(open("camera_cal/cam.p", "rb"))
mtx = pickle_data["mtx"]
dist = pickle_data["dist"]

ym_per_pix = 3.7 / 82  # meters per pixel in y dimension
xm_per_pix = 3.7 / 790  # meters per pixel in x dimension

img_size = None
#diagnostic screen. Thanks to Vivek Yadav and John Chen
diagScreen = np.zeros((960, 1280, 3), dtype=np.uint8)


#applies a mask to all channels
def apply_color_mask_hsv(image,lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    color_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR);
    return color_mask;

#applies the mask to particular channel alone
def apply_color_mask_channel(image,channel,low=180,high=255,code=cv2.COLOR_BGR2RGB):
    thresh = (low, high)
    image_conv = cv2.cvtColor(image, code)
    single_channel_image = image_conv[:,:,channel]
    binary = np.zeros_like(single_channel_image)
    binary[(single_channel_image > thresh[0]) & (single_channel_image <= thresh[1])] = 255
    return  cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)

# Define a function to threshold an image for a given range and Sobel kernel.
# The function is not used. Will experiment later
# returns a gray scale image
def dir_threshold_binary(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
# The function is not used. Will experiment later
# returns a gray scale image
def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 255),thresh=(0, np.pi/2)):
    # Convert to grayscale
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1]) & (absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
   # Return the binary image
    return binary_output

#perform bitwise addition of two images.
def bit_wise_or(img1,img2):
    new_image = cv2.bitwise_or(img1, img2)
    return new_image


# Main function that finds the left and right lane.
# The input should be thresholded left and right image
def lets_fit_and_calculate(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_warped = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)

    # Assumi1ng you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    image_mid_point = img.shape[1] / 2
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    left_inside = midpoint - leftx_base + 200;
    rightx_inside = rightx_base - midpoint;

    # Choose the number of sliding windows
    nwindows = 4
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

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

    #ignore the few top 300 pixels. So start at 300 bixels
    ploty = np.linspace(300, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # out_img = np.zeros((binary_warped.shape[0] + 200, binary_warped.shape[1], 3), np.uint8)
    # out_img = np.zeros((binary_warped.shape[0] + 200, binary_warped.shape[1], 3), np.uint8)

    #increase the image size helps to extrapolate points to the bottom of screen.
    window_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), np.uint8)
    # Color in left and right line pixels

    #to place the identified lane line accurately on the top of lane. -20 is needed since
    margin_edge = 20;

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin_edge, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + left_inside, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - rightx_inside, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx -margin_edge, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draws the lane onto the warped blank image. Draw the a green marking (inside the identified lane)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    # Draws the lane line in blue color
    cv2.polylines(window_img, np.int_([left_line_window1]), False, color=(255, 0, 0), thickness=20)
    cv2.polylines(window_img, np.int_([right_line_window2]), False, color=(255, 0, 0), thickness=20)
    result = window_img
    y_eval = np.max(ploty)
    left_curverad, right_curverad, lane_deviation = calculate_lane_curvature_and_deviation(leftx, lefty, rightx, righty,
                                                                                           y_eval, image_mid_point)
    return result, left_curverad, right_curverad, lane_deviation


# function to calculate the lane curvature and deviation
# leftx,lefty,rightx,righty - are the values for which polynomial fit needs to be performed
# The functions scale the pixels location in meters by multiplying with ym_per_pix and xm_per_pix
# y_eval - The point at which curvature needs to be calculated. Its also used to calculated the beginning of left lane and right lane
# image_mid_point - The mid point of image in pixel length in x direction
def calculate_lane_curvature_and_deviation(leftx, lefty, rightx, righty, y_eval, image_mid_point):
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # calculate the x position for y at the height of the image for left lane
    left_lane_bottom = left_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + left_fit_cr[1] * (y_eval * ym_per_pix) + \
                       left_fit_cr[2]
    # calculate the x position for y at the height of the image for right lane
    right_lane_bottom = right_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + right_fit_cr[1] * (y_eval * ym_per_pix) + \
                        right_fit_cr[2]

    # calculate the mid point of identified lane
    lane_midpoint = float(right_lane_bottom - left_lane_bottom) / 2

    lane_deviation = 0.0;
    # calculate the image center in meters from left edge of the image to right edge of image
    image_mid_point_in_meter = image_mid_point * xm_per_pix;

    # positive value indicates car is right of lane center lane, else left. Multiply by 100 to convert to centimeter
    lane_deviation = (image_mid_point_in_meter - lane_midpoint) * 100;
    return left_curverad, right_curverad, lane_deviation

#unwarp the image and transforms it perspective
def unwarp(original_src, result, left_curverad, right_curverad, lane_deviation):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_src, 'curvature right :' + str(round(right_curverad,2)) + 'm', (20, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(original_src, 'curvature left :' + str(round(left_curverad,2)) + 'm', (20, 80), font, 1, (0, 0, 0), 2)
    cv2.putText(original_src, 'lane devitation :' +  str(round(lane_deviation,2)) + 'cm', (20, 160), font, 1, (0, 0, 0), 2)
    unwarp = cv2.warpPerspective(result, MInv, (result.shape[1], result.shape[0]))
    # Combine the result with the original image
    unwarp_rz = cv2.resize(unwarp, (original_src.shape[1], original_src.shape[0]))
    result = cv2.addWeighted(original_src, 1, unwarp_rz, 0.6, 0)
    return result;

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 25.0, (1280,720))
while (cap.isOpened()):

    ret, frame = cap.read()

    img = frame;

    if (img_size == None):
        img_size = (img.shape[1], img.shape[0])

    if (frame != None):
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        warped = cv2.warpPerspective(undist, M, img_size)

        yellow_hsv_low = np.array([0, 80, 200])
        yellow_hsv_high = np.array([40, 255, 255])
        white_hsv_low = np.array([20, 0, 180])
        white_hsv_high = np.array([255, 80, 255])

        yellow_masked = apply_color_mask_hsv(warped, yellow_hsv_low, yellow_hsv_high)
        white_masked = apply_color_mask_hsv(warped, white_hsv_low, white_hsv_high)
        color_mask = apply_color_mask_channel(warped, 0, low=220, high=255, code=cv2.COLOR_BGR2RGB)

        combined_mask = bit_wise_or(yellow_masked, white_masked)
        combined_mask_2 = bit_wise_or(combined_mask, color_mask)

        result, left_curverad, right_curverad, lane_deviation = lets_fit_and_calculate(combined_mask_2)
        unwarp_image = unwarp(undist, result, left_curverad, right_curverad, lane_deviation)

        out.write(unwarp_image)

        original_resized = cv2.resize(undist, (640, 480))
        perspective_resized = cv2.resize(warped, (640, 480))
        yellow_masked_resized = cv2.resize(yellow_masked, (640, 480))
        white_masked_resized = cv2.resize(white_masked, (640, 480))
        color_mask_resized = cv2.resize(color_mask, (640, 480))
        unwarp_image_resized = cv2.resize(unwarp_image, (640, 480))
        fitted_image_resized = cv2.resize(result, (640, 480))
        combined_mask_2_resized = cv2.resize(combined_mask_2, (640, 480))

        diagScreen[0:480, 0:640] = original_resized
        diagScreen[0:480, 640:1280] = unwarp_image_resized;
        diagScreen[480:960, 0:640] = fitted_image_resized;
        diagScreen[480:960, 640:1280] = combined_mask_2_resized;
        # magnitude_gradient = mag_thresh(warped, sobel_kernel=3, mag_thresh=(30, 255),thresh=(0.4, 1.3))

        # cv2.rectangle(img, (15, 25), (200, 150), (0, 0, 255), 15)

        cv2.imshow("original", diagScreen)
        # cv2.imshow("r mask", color_mask)
        # cv2.imshow("magnitude_gradient", magnitude_gradient)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
