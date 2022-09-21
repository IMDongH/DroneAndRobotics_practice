# Drone and Robotics harris
#
# I am 201835506 임동혁

import math

import numpy as np
import cv2

if __name__ == '__main__':
    # open image
    img_rgb = cv2.imread('house.jpeg', cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    img_rgb[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('result image', img_rgb)
    cv2.waitKey(0)
