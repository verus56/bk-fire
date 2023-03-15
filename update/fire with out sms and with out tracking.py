import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('my_fire.h5')

# Initialize the video capture
video = cv2.VideoCapture(0)

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
    else:
        label = "NO FIRE"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video stream
    cv2.imshow("Capturing", frame)

    # Wait for the user to press the 'q' key to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
