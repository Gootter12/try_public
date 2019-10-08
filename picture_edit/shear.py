import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

# per表示倾斜的程度[0, 1]：图片最上面的一行像素黑色的占比
per = random.uniform(0, 0.5)

img = cv2.imread('image1.jpeg')

rows, cols = img.shape[:2]

pts1 = np.float32([[0, rows], [1, rows], [0, 0]])
pts2 = np.float32([[0, rows], [1, rows], [cols * per, 0]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('result', dst)
cv2.waitKey(0)
