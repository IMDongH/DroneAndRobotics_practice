import cv2
import numpy as np

ori_img1 = cv2.imread('set1/06.png')
ori_img2 = cv2.imread('set1/07.png')
ori_img3 = cv2.imread('set1/08.png')

img1 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(ori_img3, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(ori_img3, cv2.COLOR_BGR2GRAY)

detector = sift = cv2.xfeatures2d.SIFT_create(sigma=1.0)

keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
keypoints2, descriptor2 = detector.detectAndCompute(img2, None)
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptor1, descriptor2, 2)

ratio_thresh = 0.7
good_matches=[]
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

obj = np.empty((len(good_matches),2), dtype=np.float32)
scene = np.empty((len(good_matches),2), dtype=np.float32)
for i in range(len(good_matches)):
    obj[i,0] = keypoints1[good_matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints1[good_matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints2[good_matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints2[good_matches[i].trainIdx].pt[1]


H, _ = cv2.findHomography(scene, obj, cv2.RANSAC)

output_img = cv2.warpPerspective(ori_img1,H,
                                 (ori_img1.shape[1] + ori_img2.shape[1],
                                 max(ori_img1.shape[0], ori_img2.shape[0])))

output_img[0:ori_img1.shape[0], 0:ori_img1.shape[1]] = ori_img1
cv2.imshow("result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()