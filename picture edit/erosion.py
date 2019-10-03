import cv2
import numpy as np
import random

image = cv2.imread('image0.jpg', 1)
rows, cols = image.shape[:2]

# kern是腐蚀的唯一参数，表示核的边长
kern = random.randint(2, int(cols / 100))
cv2.imshow('origin', image)
kernel = np.ones((kern, kern), np.uint8)
erosion = cv2.erode(image, kernel, iterations=1)

cv2.imshow("ero", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()