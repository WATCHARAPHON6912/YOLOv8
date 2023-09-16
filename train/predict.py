from ultralytics import YOLO
import cv2
model = YOLO("yolov8x-seg.pt")
#show=True,hide_labels=False,save=False,hide_conf=0.5,line_thickness=2,boxes=True

img = cv2.imread('q.png')

results = model(img)
frame = results[0].plot(hide_labels=False,line_thickness=2,boxes=True)
cv2.imshow("casc",frame)
cv2.waitKey(0)