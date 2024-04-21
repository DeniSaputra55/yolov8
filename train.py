from ultralytics import YOLO 
model = YOLO('runs/detect/train/weights/best.pt') 
data = 'data/sampah-fix.mp4' 
model.predict(source=data, show=True, conf=0.5)