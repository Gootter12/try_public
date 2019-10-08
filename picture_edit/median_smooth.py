import cv2
import numpy as np
import random


def mid_smooth(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # kern是中值模糊的唯一参数，表示核的边长，要求这个参数为奇数
        kern = 0
        while kern % 2 == 0:
            kern = random.randint(2, int(cols / 30))
        # kern = int(cols/30)
        dst = cv2.medianBlur(image, kern)

        re.append(dst)
    re = np.array(re)
    return re
