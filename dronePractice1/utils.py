import cv2
# matplotlib.image 를 사용하기 위해선 matplotlib 뿐만 아니라 pillow도 깔아야 한다.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 색상 범위 설정
lower_orange = (100, 200, 200)
upper_orange = (140, 255, 255)

lower_green = (30, 80, 80)
upper_green = (70, 255, 255)

lower_blue = (0, 180, 55)
upper_blue = (20, 255, 200)

# 이미지 파일을 읽어온다
img = mpimg.imread("test.jpg", cv2.IMREAD_COLOR)

# BGR to HSV 변환
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 색상 범위를 제한하여 mask 생성
img_mask = cv2.inRange(img_hsv, lower_green, upper_green)

# 원본 이미지를 가지고 Object 추출 이미지로 생성
img_result = cv2.bitwise_and(img, img, mask=img_mask)

# 결과 이미지 생성
imgplot = plt.imshow(img_result)

plt.show()