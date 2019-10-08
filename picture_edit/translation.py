import cv2
import numpy as np
import random


def translation(images):
    re = []
    for image in images:
        rows, cols = image.shape[:2]

        # 两个参数：分别表示向左和向右移动，取值范围为[0, 1]表示移动的距离占长宽的比例
        move_right = random.uniform(0, 0.5)
        move_down = random.uniform(0, 0.5)

        rows, cols = image.shape[:2]
        # M为变换矩阵，第一个参数表示向右平移，第二个表示向下
        M = np.float32([[1, 0, move_right * cols], [0, 1, move_down * rows]])
        dst = cv2.warpAffine(image, M, (cols, rows))

        re.append(dst)
    re = np.array(re)
    return re
