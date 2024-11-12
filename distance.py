from ultralytics import YOLO 
import cv2
import cvzone
import math
import time
from cvzone.FaceMeshModule import FaceMeshDetector

# YOLO Model Setup
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Object Classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Object Widths for distance estimation
object_dimensions = {
    "cell phone": 15,
    "bottle": 35,
    "keyboard": 30.75,
    "book": 14
    
}

# Face Detection Setup
detector = FaceMeshDetector(maxFaces=1)

# Camera Setup
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # YOLO Object Detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_index = int(box.cls[0])
            class_name = classNames[class_index]

            if class_name in object_dimensions:
                # Calculate distance for detected object
                object_width = object_dimensions[class_name]
                apparent_width = w  # apparent width of the object in pixels
                distance = ((object_width * 3.9) / apparent_width) * 100
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name}, distance: {int(distance)}cm', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            else:
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Face Distance Estimation
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # Average distance between eyes in cm
        f = 668  # Approximate focal length based on known distance and object width
        d = (W * f) / w
        cvzone.putTextRect(img, f'Face Distance: ~{int(d)}cm', (face[10][1] - 75, face[10][1] - 50), scale=2)

    # FPS Calculation
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
