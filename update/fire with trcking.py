import datetime
import os

import cv2

from keras.models import load_model
import tensorflow as tf


def detect_brightness(image, radius=71):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image, maxLoc, 5, (255, 0, 0), 12)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)


def load_model(path):
    print("Loading model from disk...")
    model = tf.keras.models.load_model(path)
    print("Model loaded.")
    return model


def image_has_fire(rgb, model):
    img = rgb / 255
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)
    prediction = int(tf.round(model.predict(x=img)).numpy()[0][0])

    if prediction == 0:
        print("Fire detected!")
        return True
    else:
        print("No fire detected.")
        return False


def main():
    last_time_spoke = datetime.datetime.now()
    model_path = 'my_fire.h5'
    model = load_model(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ConnectionError("Cannot open camera")

    while True:
        ret, frame = cap.read()

        if not ret:
            raise ConnectionError("Cannot read frame from camera.")

        if image_has_fire(frame, model):
            cv2.putText(
                img=frame,
                text="Fire",
                org=(60, 60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=5
            )
            now = datetime.datetime.now()
            if (now - last_time_spoke).total_seconds() >= 5:
                os.system('spd-say "Fire detected"')
                last_time_spoke = now
            detect_brightness(frame)

        cv2.imshow('Fire Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
