import cv2
from ultralytics import YOLOWorld
from ultralytics import YOLO
print("1111")
model = YOLO('yolov8l-pose.pt')

results = model.track('bus.jpg',show=True)
annotated_frame = results[0].plot()
print(results[0].boxes.id[1])
print(results[0].keypoints.xy[1][11])
cv2.waitKey(0)
#cv2.imshow("image", annotated_frame)

