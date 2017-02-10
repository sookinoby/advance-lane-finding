import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

pickle_data = pickle.load(open("camera_cal/cam.p", "rb"))
mtx = pickle_data["mtx"]
dist = pickle_data["dist"]

img  = cv2.imread("test_images/straight_lines1.jpg",cv2.IMREAD_COLOR)

img_size = (img.shape[1], img.shape[0])
print(img_size)
src_point1 = [595,450]
src_point2 = [685,450]
src_point3 = [1022,662]
src_point4 = [295,662]


dst_point1 = [200,50]
dst_point2 = [1024,50]
dst_point3 = [1024,650]
dst_point4 = [200,650]

# For source points I'm grabbing the outer four detected corners
src = np.float32([src_point1, src_point2, src_point3, src_point4])
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result
# again, not exact, but close enough for our purposes
dst = np.float32([dst_point1, dst_point2,
                  dst_point3,
                  dst_point4])

undist = cv2.undistort(img, mtx, dist, None, mtx)

# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()

warped = cv2.warpPerspective(undist, M, img_size)
cv2.imshow("original",img)
cv2.imshow("perspective",warped)
cv2.waitKey(0)

pickle_transform = {}
pickle_transform['M'] = M;
pickle.dump(pickle_transform, open( "transform.p", "wb" ) )

print("test")

