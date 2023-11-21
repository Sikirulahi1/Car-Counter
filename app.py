import cv2 as cv
from ultralytics import YOLO
import cvzone
from sort import *
import numpy as np
import math

video_path = "../Videos/cars.mp4"
cap = cv.VideoCapture(video_path)
# cap.set(1, 620)
# cap.set(2, 480)
model = YOLO("../Yolo-Weights/yolov8n.pt")
desired_frame = (1280, 680)

totalCounts = []

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv.imread("../Practice File/mask.png")

# Adding tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

detection_threshold = 0.3

line = [360, 297, 673, 297]

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.resize(frame, desired_frame)
    mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))

    # To apply the mask to the video frame
    masked_frame = cv.bitwise_and(frame, mask)

    results = model(masked_frame, stream=True)
    detections = np.empty((0, 5))

    for result in results:

        for r in result.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = r
            score = math.ceil((score * 100)) / 100

            x1, x2, y1, y2, class_id = int(x1), int(x2), int(y1), int(y2), int(class_id)
            w, h = x2 - x1, y2 - y1
            clsNames = classNames[class_id]
            if score > detection_threshold and clsNames == "car":
                currentArray = np.array((x1, y1, x2, y2, score))
                detections = np.vstack((detections, currentArray))

            #
            # cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            # cvzone.putTextRect(frame, f"{clsNames} {score}", (max(0, x1), max(35, y1)),
            #                    scale=1, thickness=1, offset=5)

    detections = np.array(detections)
    trackerResult = tracker.update(detections)
    cv.line(frame, (line[0], line[1]), (line[2], line[3]), color=(255, 0, 255), thickness=4)
    cv.rectangle(frame, (200, 0), (350, 150), color=(255, 255, 255), thickness= -1)

    for tracks in trackerResult:
        print(tracks)
        x1, y1, x2, y2, class_id = tracks
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(frame, f"{int(class_id)}", (max(0, x1), max(35, y1)),
                           scale=1, thickness=1, offset=5)

        c1, c2 = x1 + w // 2, y1 + h // 2
        cv.circle(frame, (c1, c2), 4, (255, 0, 0), cv.FILLED)

        if line[0] < c1 < line[2] and line[1] - 15 < c2 < line[1] + 15:
            if totalCounts.count(class_id) == 0:
                totalCounts.append(class_id)
                cv.line(frame, (line[0], line[1]), (line[2], line[3]), color=(0, 255, 0), thickness=4)

    cv.putText(frame, str(len(totalCounts)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv.imshow("Window", frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
cv.destroyAllWindows()
