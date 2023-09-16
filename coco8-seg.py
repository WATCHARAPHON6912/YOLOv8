from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO(r'C:\Users\filmm\OneDrive\Desktop\YOLO\train\best.pt')  # load a custom trained

# Export the model
model.export(format='tflite')