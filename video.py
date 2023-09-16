import cv2
from ultralytics import YOLO
import torch
torch.cuda.set_device(0)
# model = YOLO('yolov8n.pt')
model = YOLO('model/yolov8x-seg.pt')
model.to('cuda')

cap = cv2.VideoCapture("Warzone.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame = results[0].plot(boxes=True)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()