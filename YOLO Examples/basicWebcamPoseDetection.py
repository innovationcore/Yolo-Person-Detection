from ultralytics import YOLO

# Load a model
model = YOLO("yolo_weights/yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model(source=0, show=True, conf=0.3, save=False)
