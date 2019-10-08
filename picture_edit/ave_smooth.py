import cv2
import numpy as np
import random


def ave_smooth(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # kern是平均模糊的唯一参数，表示核的边长
        kern = random.randint(2, int(cols / 30))
        # kern = int(cols/30)

        dst = cv2.blur(image, (kern, kern))
        re.append(dst)

    return re
