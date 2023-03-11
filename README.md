# bk-fire
lien de model : https://drive.google.com/drive/folders/19TqQx8erd-fHEjRjUrXOeXKocdzXeenr?usp=share_link


Before you run this project,;
```
firewithtsms.py  just for test model with webcam
fire-with-all.py  detect fire and send sms for user
robot.py  sytem for robot with our ai model
```

The OPEN FIRE Python scripts demonstrate the implementation of two distinct systems for fire detection: a Convolutional Neural Network (CNN) model for image classification and a robotic system capable of detecting and responding to fires in real-time.

The CNN model is designed for binary classification of images into two classes: "fire" and "no fire." The script leverages the Keras ImageDataGenerator class to preprocess and augment the images stored in the "train" and "valid" directories before feeding them to the model. The CNN architecture comprises convolutional layers, followed by max-pooling layers, and a fully connected layer with a Rectified Linear Unit (ReLU) activation function. Dropout layers are incorporated to prevent overfitting, and the output layer is a dense layer with a softmax activation function for binary classification. The model is compiled with categorical_crossentropy loss function and Adam optimizer with a learning rate. During training, the model is evaluated based on accuracy as the metric. After training, the model's performance is evaluated by plotting accuracy and loss over the epochs.

The second system is a robotic system designed to detect and respond to fires in real-time. The system utilizes various sensors and actuators, including flame and smoke sensors, a buzzer, LED, water pump, and a servo motor for a robotic arm. The system uses the Raspberry Pi GPIO pins to interface with the sensors and actuators and the PiCamera module to capture images of the environment, which are then preprocessed and used to make predictions using the trained machine learning model to determine if a fire is present.

The implementation of the robotic system also includes the use of libraries such as OpenCV, NumPy, Pillow, TensorFlow, Keras, Twilio, and Geocoder. OpenCV is an open-source computer vision library used for image and video processing tasks. NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays. Pillow is a library that adds support for opening, manipulating, and saving many different image file formats. TensorFlow is an open-source machine learning library developed by Google that is used for dataflow and differentiable programming across a range of tasks. Keras is a high-level neural networks API written in Python that runs on top of TensorFlow, CNTK, or Theano. Twilio is a cloud communications platform that provides APIs for sending and receiving SMS, voice, and video messages. Geocoder is a library for Python that provides a geocoding interface to several popular APIs, including Google Maps, Bing Maps, OpenStreetMap, and more.

Overall, the provided Python scripts provide a useful framework for building a fire detection and response system
