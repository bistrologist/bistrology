import cv2
import numpy as np


imgPath = "/Users/scottwillis/Documents/Coding/Image Processing/bistrology-git/bistrology/resources/circle1.jpeg"
image = cv2.imread(imgPath)
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurimage = cv2.medianBlur(grayimage, 5)

cv2.imwrite('circle_orig.jpg', image)
cv2.imwrite('circle_bw.jpg', grayimage)
cv2.imwrite('circle_blur.jpg', blurimage)

"""
Parameters:
image – 8-bit, single-channel, grayscale input image.
circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x, y, radius) .
circle_storage – In C function this is a memory storage that will contain the output sequence of found circles.
method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
minRadius – Minimum circle radius.
maxRadius – Maximum circle radius.
param2 is most important = higher number is less circles
circles = cv2.HoughCircles(image=blurimage, method=cv2.HOUGH_GRADIENT, dp=1, minDist=200, param1=50, param2=30, minRadius=100, maxRadius=200)
circles = cv2.HoughCircles(image=blurimage, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=50, minRadius=100, maxRadius=200)
"""

circles = cv2.HoughCircles(image=blurimage, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=50, minRadius=100, maxRadius=200)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(blurimage, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(blurimage, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imwrite('circle_processed.jpg', blurimage)
