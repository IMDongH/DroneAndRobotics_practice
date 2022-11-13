import cv2 as cv
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("./model_data_yolov5n.pt", map_location=device)


img = Image.open("./img.png")
results = model(img)
results2 = results.pandas().xyxy[0][['name', 'xmin', 'ymin', 'xmax', 'ymax']]

for num, i in enumerate(results2.values):
        cv.putText(img, i[0], (int(i[1]), int(i[2])), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv.rectangle(img, (int(i[1]), int(i[2])), (int(i[3]), int(i[4])), (0, 0, 255), 3)
        break
cv.imshow('temp', img)
cv.waitKey(1)