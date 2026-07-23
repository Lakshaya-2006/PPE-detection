from ultralytics import YOLO
print("YOLO imported successfully")
model=YOLO("yolov8n.pt")
print("model has been imported")

results=model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
    project="runs",
    name="PPE_Detection",
    pretrained=True,
    verbose=True
)
print("MODEL TRAINED SUCCESSFULLYY!!!")