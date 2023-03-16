import cv2 as cv
from PIL import Image
import torch
import requests

model = torch.hub.load('ultralytics/yolov5', 'custom', 'openfiree.pt')


def obti(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)
    results = model(im)
    results.render()
    detections = results.pandas().xyxy[0]

    return detections


# Twilio credentials
account_sid = 'C9553c9e172fb190b011b66efd1313813'
auth_token = 'a41a92bc13b3009dc3e52da82cbbd00b'
from_number = '+15673671688'
to_number = '+213559407669'

# Define variables for fire detection
extinguished = True
scanning = False
i = 0

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    detections = obti(pil_image)
    if not detections.empty:
        print("Fire detected")
        if extinguished and not scanning:
            scanning = True
            i = 0
            scanned_frames = 0
        if extinguished and scanning and scanned_frames < 1000:
            if not detections.empty:
                i += 1
            scanned_frames += 1
        if extinguished and scanning and scanned_frames == 1000:
            if i / 30 > 1:
                extinguished = False
                scanning = False
                message = f"Fire detected for more than 30 frames. Please check the location."
                data = {
                    'Body': message,
                    'From': from_number,
                    'To': to_number
                }
                response = requests.post(
                    f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json',
                    auth=(account_sid, auth_token),
                    data=data
                )
                print("SMS sent.")
    else:
        print("No fire detected")
        if not extinguished and not scanning:
            scanning = True
            i = 0
            scanned_frames = 0
        if not extinguished and scanning and scanned_frames < 3000:
            if detections.empty:
                i += 1
            scanned_frames += 1
        if not extinguished and scanning and scanned_frames == 3000:
            if i / 3000 == 1:
                extinguished = True
                scanning = False
                print("Fire extinguished.")

    for _, detection in detections.iterrows():
        x1, y1 = int(detection['xmin']), int(detection['ymin'])
        x2, y2 = int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        label = f'Fire {conf:.2f}'
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0 , 255, 0), 2)
        x_pos = int((x1 + x2) / 2)
        y_pos = int((y1 + y2) / 2)
        distance = round((x2 - x1) * 2.54 / 96, 2)
        cv.putText(frame, f"Position: ({x_pos}, {y_pos})", (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(frame, f"Distance: {distance} cm", (x1, y2 + 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if extinguished == True:
            if i == 30:
                # Send SMS code here
                message = f"Fire detected for 30 frames. Please check the location."
                data = {
                    'Body': message,
                    'From': from_number,
                    'To': to_number
                }
                response = requests.post(
                    f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json',
                    auth=(account_sid, auth_token),
                    data=data
                )
                print("SMS sent.")
                extinguished = False
        else:
            if i == 3000:
                extinguished = True
                print("Fire has been extinguished for 3000 frames.")

        cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
