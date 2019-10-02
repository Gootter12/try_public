import numpy as np
import argparse
import imutils
import cv2
import random

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("begin")
for image in x_train:

    # angle 表示图片旋转的角度
    angle = random.randint(0, 90)

    # cv2.imshow("Rotated", image)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    # cv2.imshow("Rotated by Degrees", rotated)
