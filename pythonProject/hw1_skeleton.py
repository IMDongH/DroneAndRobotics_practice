#
# Drone and Robotics Homework #1
#
# I am 201835506 임동혁
#
import math

import numpy as np
import cv2
import numpy as np


def convertRGBtoGray(rgb):
    gray = rgb[:, :, 0] * 0.114 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.299
    gray = np.squeeze(gray).astype(np.uint8)
    return gray


def brightness(gray, val):
    res_img = np.clip((gray.copy() + val), 0, 255).astype(np.uint8)

    return res_img


def contrast(gray, grad, inter):
    res_img = np.clip((grad * gray.copy() + inter), 0, 255).astype(np.uint8)

    return res_img


def scaling1(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h * s), int(w * s)), gray.dtype)

    #  forward warping
    for r in range(h):
        for c in range(w):
            res_img[r * s][c * s] = gray[r][c]

    return res_img


def scaling2(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h * s), int(w * s)), gray.dtype)

    #  backward warping
    for r in range(h * s):
        for c in range(w * s):
            res_img[r][c] = gray[math.floor(r / s)][math.floor(c / s)]

    return res_img


# deg: angle
def rotation(gray, angle):
    h, w = gray.shape
    print(h, w)
    res_img = np.zeros((int(h), int(w)), gray.dtype)
    angle = angle * math.pi / 180

    affine = np.array([[math.cos(angle), -1 * math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])

    for r in range(h):
        for c in range(w):
            data = np.matmul(affine, np.array([[r - h / 2], [c - w / 2]])) + np.array([[h / 2], [w / 2]])
            if h > math.ceil(data[0]) > 0 and 0 < math.ceil(data[1]) < w:
                res_img[r][c] = gray[math.ceil(data[0])][math.ceil(data[1])]

    return res_img


if __name__ == '__main__':
    # open image
    img_rgb = cv2.imread('image.png', cv2.IMREAD_COLOR)

    # get dimension
    h, w, ch = img_rgb.shape

    # if you want to know the dimension of img_rgb, remove comment below
    # print(img_rgb.shape)

    # mission 1 : convert color image to grayscale
    img_gray = convertRGBtoGray(img_rgb)

    # mission 2: decrease brightness
    # caution: clip values between 0 ~ 255
    img_bright = brightness(img_gray, 50.)

    # mission 3: decrease brightness
    # a: gradient, b:an intercept of y axis
    img_contrast = contrast(img_gray, 1.5, -50.)

    # mission 4: scaling
    # move source pixels to target
    img_scaling1 = scaling1(img_gray, 3)

    # mission 5: scaling2
    # move source pixels to target
    img_scaling2 = scaling2(img_gray, 3)

    # mission 6: rotation
    # caution: Rotate the image around the center of the image.
    img_rotation = rotation(img_gray, 30)

    # concatenate results
    img_res1 = cv2.hconcat([img_gray, img_bright, img_contrast])
    img_res2 = cv2.hconcat([img_scaling1, img_scaling2])

    # display input image & results
    cv2.imshow('input image', img_rgb)
    cv2.imshow('gray, bright, contrast', img_res1)
    cv2.imshow('scaling', img_res2)
    cv2.imshow('rotation', img_rotation)
    cv2.waitKey(0)
