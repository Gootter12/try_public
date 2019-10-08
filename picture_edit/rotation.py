import numpy as np
import argparse
import imutils
import cv2
import random


def rotation(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # angle 表示图片旋转的角度
        angle = random.randint(0, 90)

        # cv2.imshow("Rotated", image)

        center = (cols // 2, rows // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        dst = cv2.warpAffine(image, M, (cols, rows))

        re.append(dst)
    re = np.array(re)
    return re
