import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def shear(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # per表示倾斜的程度[0, 1]：图片最上面的一行像素黑色的占比
        per = random.uniform(0, 0.5)

        pts1 = np.float32([[0, rows], [1, rows], [0, 0]])
        pts2 = np.float32([[0, rows], [1, rows], [cols * per, 0]])

        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (cols, rows))

        re.append(dst)
    re = np.array(re)
    return re
