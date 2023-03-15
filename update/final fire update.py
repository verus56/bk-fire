import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from twilio.rest import Client

# Load the saved model
model = tf.keras.models.load_model('my_fire.h5')

# Initialize the video capture
video = cv2.VideoCapture(0)

# Initialize Twilio client
account_sid = 'AC9553c9e172fb190b011b66efd1313813'
auth_token = 'a41a92bc13b3009dc3e52da82cbbd00b'
client = Client(account_sid, auth_token)


# Create a function to send SMS message
def send_sms(message):
    client.messages.create(
        to='+213559407669',
        from_='+15673671688',
        body=message)


# Initialize counters
frame_count = 0
sms_count = 0

while True:
    # Read the next frame
    _, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resize the image to 224x224 because we trained the model with this image size
    im = im.resize((224, 224))

    # Convert the image to an array and normalize the pixel values
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255

    # Predict the class probabilities for the image
    probabilities = model.predict(img_array)[0]

    # Get the class label with the highest probability
    prediction = np.argmax(probabilities)

    # Add a text label indicating whether there is fire or not
    if prediction == 0:
        label = "FIRE"
        frame_count += 1
        if frame_count % 30 == 0 and sms_count < 2:
            message = "Fire detected!"
            send_sms(message)
            sms_count += 1
    else:
        label = "NO FIRE"
        frame_count = 0  # Reset frame count when no fire is detected

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video stream
    cv2.imshow("Capturing", frame)

    # Wait for the user to press the 'q' key to quit
    key = cv2.waitKey(1)
    if key == ord('q') or sms_count == 2:
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
