import cv2
import numpy as np
import random

image = cv2.imread('image2.jpg')
rows, cols = image.shape[:2]

# kern是中值模糊的唯一参数，表示核的边长，要求这个参数为奇数
kern = 0
while kern % 2 == 0:
    kern = random.randint(2, int(cols / 30))
# kern = int(cols/30)

cv2.imshow('origin', image)
dst = cv2.medianBlur(image, kern)
cv2.imshow("ave", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
