
---

#Advanced Lane Finding Project

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


##1. Camera Calibration.

Cameras calibration is required for correcting distortion of the image caused by the design of camera lens. Although cameras are designed with high precision, the lens can still be little deformed due to temperature, environment, pressure and etc. This will cause light rays to bend by varying degree when they hit the lens. Also, the alignment of the sensor with the camera lens may not be parallel. To correct the image for distortion caused by the lens is one of basic and important step we need to perform before we process the image.

We started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here we are assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  These objects point are mapped onto the image coordinates. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. This function returns distortion coefficients and camera matrix(focal length, image center) along with the position of each chessboard pattern in camera coordinate system. A detailed discussion of this technique can be found in [Microsoft's research] (http://research.microsoft.com/en-us/um/people/zhang/calib/)

Once the distortion correction and camera matrix were calculated, we used it to correct the distorted image using OpenCV's undistort() function. The output is shown below. Since the distortion is not clearly visible to human eyes, I have shown a difference image below. The camera matrix (mtx) and distortion(dist) coefficients were stored as a file, as the remain constant if the camera's lens/focal length remains the same.

The code for this is contained in calibrate.py

![Alt text](output_images/distortion-correction.png?raw=true "camera calibration")


##2. Perspective transforms:

We decided to perform perspective transforms before any filtering, since perspective transform will remove unwanted part of the images and focus on the road.
When an image is captured by a camera, the objects which are farther away looks smaller in an image. This causes the parallel lines in the real world to appear as if they intersect in images. In order to undo such an effect, we need to transform the image into a perspective free image. Perspective transform also helps to view a scene from the different viewpoint (as if the camera is moved some other position and the picture was taken). So we decided to view the road from a bird-eye point of view. For performing perspective transform we selected four points in an original undistorted image and then mapped it to a set of four points which forms a rectangle.


```python
#The four points in source are
src_point1 = [595,450]
src_point2 = [685,450]
src_point3 = [1022,662]
src_point4 = [295,662]

#The four points in destination are
dst_point1 = [200,50]
dst_point2 = [1024,50]
dst_point3 = [1024,720]
dst_point4 = [200,720]

src = np.float32([src_point1, src_point2, src_point3, src_point4])
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result
# again, not exact, but close enough for our purposes
dst = np.float32([dst_point1, dst_point2,
                  dst_point3,
                  dst_point4])
 ```


We used getPerspectiveTransform(src, dst) to get the perspective matrix (M). Then we use the matrix M and warpPerspective(undist, M, img_size) to compute the perspective transformed image. The img_size should be in a column, row format due to porting from cv to cv2. Since the perspective transform matrix is constant for a particular camera at the particular position, we save the matrix M.

The code for this is contained in perspective.py

![Alt text](output_images/prespective_transformed.png?raw=true "camera calibration")


##3.Pipeline for Identifying lanes (single images)

The lane identifying pipeline consists of series of image transformation that will help to identify the lane lines, curvature, and vehicle position with respect to the center of the lane. Once the frame(image) is obtained to the video (can be live stream), the image undergo the following transformation and filtering

###1. Distortion correction:

The image is corrected for distortion using the camera matrix(mtx) , distortion(dist) coeffiecients and undistort() function.


###2. Perspective transform:

The image is corrected for perspective and a bird-eye view of image is obtained using the perspective transform matrix (M) and OpenCV's warpPerspective()



###3. Yellow lane marking:

The image was transformed to HSV color space from BGR since HSV can be used to extract particular color component with varying lighting conditions. We used the following yellow color mask range to extract the yellow lane:

```python
    yellow_hsv_low = np.array([0, 80, 200])
    yellow_hsv_high = np.array([40, 255, 255])
```
output

![Alt text](output_images/yellow_masked.jpg?raw=true, "Yellow mask")

###4. White lane marking:


The image was transformed to HSV color space from BGR since HSV can be used to extract particular color component with varying lighting condition.We used the following white color mask range to extract the white lane:

```python
    white_hsv_low = np.array([20, 0, 180])
    white_hsv_high = np.array([255, 80, 255])
```
output
![Alt text](output_images/white_masked.jpg?raw=true, "White mask")

###5. Red channel mask:

Although the above two mask were able to identify the yellow and white line at most places, it misses identifying those lines in few parts. To solve this issue we used the mask on R channel of the image. 

```python
    R_low=220, R_high=255
```

output
![Alt text](output_images/r_masked.jpg?raw=true, "r channel mask")

###6. Combine masking:

Since each mask applies a particular filter to the image, we combine them together to achieve the desired output. Below the function to combine two images. 

```python
   def bit_wise_or(img1,img2):
    new_image = cv2.bitwise_or(img1, img2)
    return new_image
```

output
![Alt text](output_images/combined_masked.png?raw=true, "combined mask")

###7. identifying left and right lane:

The above filtering helped us to identify the lanes lines with reasonable accuracy. After the lane lines are identified we used to identify the left lane and right lane by creating a histogram over the bottom half of the image. The two peaks of the histogram represent the left lane and right lane respectively. Once the starting point of the each lane is identified in the x direction, we use a sliding window from the bottom of the image to middle half of the image to identify all the non-zero pixel (lane pixel). Then we fit second order polynomial through those points, to identify the lane and its direction using OpenCv's polyfit function. Below is the visualization of histogram

Visualization - Picture from udacity course website
![Alt text](output_images/histogram.png?raw=true, "Histogram visualization")


###8.The curvature of the road:

The polynomial fit will return an equation as followed.
f(y)=Ay**2+By+C 

To calculate the curvature we calculate the following for left lane and right lane
![Alt text](output_images/curvarture.png?raw=true, "curvature")

Before we calculate the curvature, we need to convert the pixels to world coordinate, we use the following conversion factor (measured manually from the image and USA road specification)

```python
    ym_per_pix = 3.7 / 82  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 790  # meters per pixel in x dimension
```
Then we fit and measure the curvature as follows.

```python
   left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
```
![Alt text](output_images/result_p_transformed.jpg?raw=true, "lane detected")


###9. Car offset from the middle of the lane:
The car's camera can be assumed to be mounted in the center of the car. Calculating the center of the image will give the car position in the image. The difference between the calculated right and left lane divided by the 2 gives the midpoint of the lane. The difference between the center of the image and midpoint of the lane gives the estimate of the car offset from the center.

```python
    
    #calculate the x position for y at the height of the image for left lane
    left_lane_bottom = left_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + left_fit_cr[1] * (y_eval * ym_per_pix) + \
                       left_fit_cr[2]

    #calculate the x position for y at the height of the image for right lane                   
    right_lane_bottom = right_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + right_fit_cr[1] * (y_eval * ym_per_pix) + \
                        right_fit_cr[2]

    #calculate the mid point of identified lane
    lane_midpoint = float(right_lane_bottom - left_lane_bottom) / 2

    lane_deviation = 0.0;
    #calculate the image center in meters from left edge of the image to right edge of image
    image_mid_point_in_meter = image_mid_point * xm_per_pix;

    # positive value indicates car is right of lane center lane, else left. Multiply by 100 to convert to centimeter
    lane_deviation = (image_mid_point_in_meter - lane_midpoint) * 100;
```

###10. Unwarping the image back:
Since all the calculation were performed from bird's-eye view, we need to change it back into perspective transformed image. For this, we need to calculate the inverse of transformation matrix (MInv). We combine the originally distorted image with transformed detected lane image to obtain the final output.


```python
def unwarp(original_src,result):
    unwarp = cv2.warpPerspective(result, MInv, (result.shape[1], result.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original_src, 1, unwarp, 0.6, 0)
    return result;

```
![Alt text](output_images/unwarp_image_tranformed.jpg?raw=true, "final output")


###Final Output
Here is the link to my final output. 
Here's a [link to my video result](./project_video.mp4)

The output.avi files contains the annotated video

The code for this is contained in play.py

The output for the first track is 
[![Output](output_images/youtube.png?raw=true)](https://youtu.be/HpgY4iph8gg)

#Issues

The methods we use depends heavily on the color of the lane lines (yellow, white). It might not capture the lanes pixels in all condition. Also, it captures anything in yellow or white( other lane marking). Other lane lines marking car be removed partially by filtering the direction of the gradient.  We can adaptively adjust the threshold to capture the lane lines alone for the varying condition. 
