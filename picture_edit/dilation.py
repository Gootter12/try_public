import cv2
import numpy as np
import random


def dilation(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # kern是膨胀的唯一参数，表示核的边长
        kern = random.randint(2, int(cols / 100))
        kernel = np.ones((kern, kern), np.uint8)
        dst = cv2.dilate(image, kernel, iterations=1)

        re.append(dst)
    re = np.array(re)
    return re
