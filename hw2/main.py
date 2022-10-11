import numpy as np
import cv2 as cv

FLANN_INDEX_LSH = 6


def main(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))

    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])

    flann_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)  # 2

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            matches.append((m[0].trainIdx, m[0].queryIdx))


    keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
    keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])

    H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC, 4.0)



    result = cv.warpPerspective(img1, H,
                                (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result


if __name__ == '__main__':
    img1 = cv.imread('set1/08.png')
    img2 = cv.imread('set1/07.png')
    img3 = cv.imread('set1/06.png')

    result = main(img1,img2)
    result = main(result, img3)

    pts_src = np.array([[0.0, 0.0], [1090.0, 0.0], [0.0, 800.0], [1250.0, 800.0]])
    pts_dst = np.array([[0.0, 0.0], [1100.0, 0.0], [0.0, 800.0], [1200.0, 800.0]])

    im_dst = np.zeros((800, 1150, 3), np.uint8)

    h, status = cv.findHomography(pts_src, pts_dst)

    im_out = cv.warpPerspective(result, h, (im_dst.shape[1], im_dst.shape[0]))

    cv.imshow('result2', im_out)
    cv.waitKey()
    cv.destroyAllWindows()