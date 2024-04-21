from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("model/yolov8m.pt")

cap = cv2.VideoCapture("data/sampah-fix.mp4")
assert cap.isOpened()
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_of_interest = [(20, 600), (1700, 604), (1700, 560), (20,560)]

video_writer = cv2.VideoWriter("hasil training.mp4",
cv2.VideoWriter_fourcc(*'mp4v'),
fps,
(w, h))

while cap.isOpened():
    succes, im0 = cap.read()
    if not succes:
        print("Video frame is empty or video processing has been succesfullyt completed.")
        break
        tracks = model.tracks(im0, persist=True, show=False)
        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)
