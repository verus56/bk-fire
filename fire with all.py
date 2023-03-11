import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from twilio.rest import Client
import geocoder

# Load the saved model
model = tf.keras.models.load_model('my_fire.h5')

# Initialize the video capture
video = cv2.VideoCapture(0)

# Initialize Twilio client
account_sid = 'AC9553c9e172fb190b011b66efd1313813'
auth_token = 'a41a92bc13b3009dc3e52da82cbbd00b'
client = Client(account_sid, auth_token)

# Create a function to send SMS message
def send_sms(message, latitude, longitude):
    # Construct the Google Maps URL
    maps_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

    # Create the SMS message with the Google Maps link
    sms_message = f"{message}\nLocation: {maps_url}"

    client.messages.create(
        to='+213559407669',
        from_='+15673671688',
        body=sms_message)

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
        message = "Fire detected!"
        g = geocoder.ip('me')
        latlng = g.latlng
        send_sms(message, latlng[0], latlng[1])
    else:
        label = "NO FIRE"

    # Add the label and location information to the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    g = geocoder.ip('me')
    latlng = g.latlng
    cv2.putText(frame, f"Location: {latlng}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video stream
    cv2.imshow("Capturing", frame)

    # Wait for the user to press the 'q' key to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
