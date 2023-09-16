
from ultralytics import YOLO
import cv2
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')

# Run inference on 'bus.jpg' with arguments
# model.predict('cycle.jpg', save=True, conf=0.5,)
img = cv2.imread('cycle.jpg')
results = model.predict(img, stream=True)                 # run prediction on img
print(results)
for result in results:                                         # iterate results
    boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
    for box in boxes:                                          # iterate boxes
        r = box.xyxy[0].astype(int)                            # get corner points as int
        print(r)                                               # print boxes
        cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 2)   # draw boxes on img
cv2.imshow("",img)
cv2.waitKey(0)