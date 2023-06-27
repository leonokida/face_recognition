from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")
#model = YOLO("fromscratch.pt")

# Use the model
results = model.predict(source="0", show=True, conf=0.5)
print(results)