import cv2 as cv
from PIL import Image
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', 'openfiree.pt')

def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)
    results = model(im)
    results.render()
    detections = results.pandas().xyxy[0]

    return detections

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    detections = yolo(pil_image)
    if not detections.empty:
        print("Fire detected")
        for _, detection in detections.iterrows():
            x1, y1 = int(detection['xmin']), int(detection['ymin'])
            x2, y2 = int(detection['xmax']), int(detection['ymax'])
            conf = detection['confidence']
            label = f'Fire {conf:.2f}'
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            x_pos = int((x1 + x2) / 2)
            y_pos = int((y1 + y2) / 2)
            distance = round((x2 - x1) * 2.54 / 96, 2)
            cv.putText(frame, f"Position: ({x_pos}, {y_pos})", (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(frame, f"Distance: {distance} cm", (x1, y2 + 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else :
        print("No Fire detected")

    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
