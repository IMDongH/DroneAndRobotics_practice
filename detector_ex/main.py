import numpy as np
import cv2

def brightness(gray, val):
    res_img = gray.copy()

    val += 0.0
    res_img = gray + val
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    return res_img

def contrast(gray, grad, inter):
    res_img = gray.copy() * 0

    # your code here
    res_img = gray*grad+inter
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    return res_img

img1 = cv2.imread('db/object1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('db/scene2.jpg', cv2.IMREAD_GRAYSCALE)

img1 =  contrast(img1, 1.3, -60)
# create(nfeatrues, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
detector = sift = cv2.xfeatures2d.SIFT_create(sigma=1.0)

keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
keypoints2, descriptor2 = detector.detectAndCompute(img2, None)

img_res_detect1 = cv2.drawKeypoints(img1, keypoints1, None)
img_res_detect2 = cv2.drawKeypoints(img1, keypoints1, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptor1, descriptor2, 2)

ratio_thresh = 0.6
good_matches=[]
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty( (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8 )
cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img_res_detect1', img_res_detect1)
# cv2.imshow('img_res_detect2', img_res_detect1)
# cv2.imshow('img_matches', img_matches)
# cv2.waitKey(0)

obj = np.empty((len(good_matches),2), dtype=np.float32)
scene = np.empty((len(good_matches),2), dtype=np.float32)
for i in range(len(good_matches)):
    obj[i,0] = keypoints1[good_matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints1[good_matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints2[good_matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints2[good_matches[i].trainIdx].pt[1]

H,_ = cv2.findHomography(obj,scene,cv2.RANSAC)

obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = img1.shape[1]
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = img1.shape[1]
obj_corners[2,0,1] = img1.shape[0]
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = img1.shape[0]

scene_corners = cv2.perspectiveTransform(obj_corners,H)

img_obj_scene = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
scene_corners = cv2.perspectiveTransform(obj_corners, H)

img_obj_scene = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.line(img_obj_scene, (int(scene_corners[0,0,0]),int(scene_corners[0,0,1])),
(int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), (255,0,0), 3)
cv2.line(img_obj_scene, (int(scene_corners[1,0,0]),int(scene_corners[1,0,1])),
(int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), (255,0,0), 3)
cv2.line(img_obj_scene, (int(scene_corners[2,0,0]),int(scene_corners[2,0,1])),
(int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), (255,0,0), 3)
cv2.line(img_obj_scene, (int(scene_corners[3,0,0]),int(scene_corners[3,0,1])),
(int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), (255,0,0), 3)

cv2.imshow('img_obj_scene', img_obj_scene)

cv2.waitKey(0)