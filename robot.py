import RPi.GPIO as GPIO
import time
import os
from picamera import PiCamera
from tensorflow import keras
import numpy as np

# Set up GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define constants for pins
flame_sensor_pin = 18
smoke_sensor_pin = 23
buzzer_pin = 24
led_pin = 25
pump_pin_A = 12
pump_pin_B = 16
motor_pin_A1 = 20
motor_pin_A2 = 21
motor_pin_B1 = 26
motor_pin_B2 = 19

# Initialize servo motor
servo_pin = 4
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)
servo.start(0)

# Set up camera
camera = PiCamera()
camera.resolution = (224, 224)

# Load trained model
model = keras.models.load_model('my_fire.h5')

def read_flame_sensor():
    # Read value from flame sensor
    GPIO.setup(flame_sensor_pin, GPIO.IN)
    flame_sensor_value = GPIO.input(flame_sensor_pin)

    return flame_sensor_value


def read_smoke_sensor():
    # Read value from smoke sensor
    GPIO.setup(smoke_sensor_pin, GPIO.IN)
    smoke_sensor_value = GPIO.input(smoke_sensor_pin)

    return smoke_sensor_value


def sound_alarm():
    # Turn on LED and sound buzzer
    GPIO.output(led_pin, GPIO.HIGH)
    GPIO.output(buzzer_pin, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(led_pin, GPIO.LOW)
    GPIO.output(buzzer_pin, GPIO.LOW)


def move_towards_fire():
    # Move robot towards fire
    GPIO.output(motor_pin_A1, GPIO.HIGH)
    GPIO.output(motor_pin_B1, GPIO.HIGH)
    GPIO.output(motor_pin_A2, GPIO.LOW)
    GPIO.output(motor_pin_B2, GPIO.LOW)


def move_away_from_fire():
    # Move robot away from fire
    GPIO.output(motor_pin_A1, GPIO.LOW)
    GPIO.output(motor_pin_B1, GPIO.LOW)
    GPIO.output(motor_pin_A2, GPIO.LOW)
    GPIO.output(motor_pin_B2, GPIO.LOW)


def activate_water_pump():
    # Activate water pump
    GPIO.output(pump_pin_A, GPIO.HIGH)
    GPIO.output(pump_pin_B, GPIO.LOW)


def deactivate_water_pump():
    # Deactivate water pump
    GPIO.output(pump_pin_A, GPIO.LOW)
    GPIO.output(pump_pin_B, GPIO.LOW)


def move_servo_arm(angle):
    # Move servo arm to specified angle
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    servo.ChangeDutyCycle(0)


def detect_fire():
    # Capture image with camera
    camera.start_preview()
    time.sleep(2)
    camera.capture('image.jpg')
    camera.stop_preview()

    # Preprocess image for model input
    img = keras.preprocessing.image.load_img(
        'image.jpg', target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize pixel values

    # Use the trained model to make predictions
    predictions = model.predict(img_array)
    fire_probability = predictions[0][0]

    # Check if fire is detected
    if fire_probability > 0.5:
        return True
    else:
        return False

class Robot:

    def detect_fire():
        # Capture image with camera
        camera.start_preview()
        time.sleep(2)
        camera.capture('image.jpg')
        camera.stop_preview()

        # Preprocess image for model input
        img = keras.preprocessing.image.load_img(
            'image.jpg', target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction with model
        prediction = model.predict(img_array)
        fire_probability = prediction[0][1]  # probability that fire is present

        # Determine whether fire is present based on threshold
        threshold = 0.5  # adjust this as needed
        fire_detected = (fire_probability >= threshold)

        return fire_detected

    def activate_firefighting_mode(self):
        # Perform firefighting actions
        if self.read_flame_sensor() or self.read_smoke_sensor():
            self.sound_alarm()
            self.move_towards_fire()
            self.activate_water_pump()
            self.move_servo_arm(90)
        else:
            self.deactivate_water_pump()
            self.move_servo_arm(0)

    def activate_wander_mode(self):
        # Wander around
        self.move_away_from_fire()
        self.deactivate_water_pump()
        self.move_servo_arm(0)




def main():
    # Initialize the robot
    robot = Robot()

    # Main loop
    while True:
        # Check for fire
        if detect_fire(robot):
            # Perform firefighting actions
            robot.activate_firefighting_mode()
        else:
            # Wander around
            robot.activate_wander_mode()


if __name__ == '__main__':
    main()

