import cv2
import numpy as np
import random


def erosion(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # kern是腐蚀的唯一参数，表示核的边长
        kern = random.randint(2, int(cols / 100))
        kernel = np.ones((kern, kern), np.uint8)
        dst = cv2.erode(image, kernel, iterations=1)

        re.append(dst)
    re = np.array(re)
    return re
