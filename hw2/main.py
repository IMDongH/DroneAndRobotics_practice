# import numpy as np
# import cv2 as cv
#
# FLANN_INDEX_LSH = 6
#
#
# def main(img1, img2):
#     gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#     gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
#
#     detector = cv.BRISK_create()
#     keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
#     keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
#     print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))
#
#     keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
#     keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
#
#     flann_params = dict(algorithm=FLANN_INDEX_LSH,
#                         table_number=6,  # 12
#                         key_size=12,  # 20
#                         multi_probe_level=1)  # 2
#
#     matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
#     raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)  # 2
#
#     matches = []
#     for m in raw_matches:
#         if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
#             matches.append((m[0].trainIdx, m[0].queryIdx))
#
#
#     keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
#     keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])
#
#     H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC, 4.0)
#
#
#
#     result = cv.warpPerspective(img1, H,
#                                 (img1.shape[1] + img2.shape[1], img1.shape[0]))
#     result[0:img2.shape[0], 0:img2.shape[1]] = img2
#     return result
#
#
# if __name__ == '__main__':
#     img1 = cv.imread('set1/08.png')
#     img2 = cv.imread('set1/07.png')
#     img3 = cv.imread('set1/06.png')
#
#     # img1 = cv.imread('set2/004.png')
#     # img2 = cv.imread('set2/003.png')
#     # img3 = cv.imread('set2/002.png')
#     # img4 = cv.imread('set2/001.png')
#
#     result = main(img1,img2)
#     result = main(result, img3)
#     # result = main(result, img4)
#     cv.imshow('result', result)
#     cv.waitKey()
#     cv.destroyAllWindows()

import cv2
cap = cv2.VideoCapture('video/horse.avi')
tmpl = cv2.imread('video/horse_template.jpg', 0)
while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    match = cv2.matchTemplate(gray, tmpl, cv2.TM_SQDIFF_NORMED)
    _, _, min_pos, _ = cv2.minMaxLoc(match)
    tmpl = gray[min_pos[1]:min_pos[1]+tmpl.shape[0], min_pos[0]:min_pos[0]+tmpl.shape[1]]
    cv2.rectangle(frame, min_pos, (min_pos[0] + tmpl.shape[1], min_pos[1] + tmpl.shape[0]), (255, 0, 0), 2)
    cv2.imshow('scene', frame)
    cv2.imshow('template', tmpl)
    ch = cv2.waitKey(30)
    if ch == 27: break