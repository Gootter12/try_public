import cv2
import numpy as np
from PIL import Image
import random

# 两个参数用于控制缩放，分别控制长宽两个维度，取值范围[0, 1]，表示缩小后的长宽占原来的比例
scale_width = random.uniform(0.5, 1)
scale_height = random.uniform(0.5, 1)

img = cv2.imread('image1.jpeg', 1)
rows, cols = img.shape[:2]
res1 = cv2.resize(img, None, fx=scale_width, fy=scale_height, interpolation=cv2.INTER_CUBIC)

height, width, mode = res1.shape
res2 = np.zeros(img.shape, np.uint8)

print(res1.shape)
print(res2.shape)
res2[0:height, 0:width] = res1
cv2.imshow('img2', res2)

cv2.waitKey(0)
cv2.destroyAllWindows()
