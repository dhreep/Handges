# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run gesture recognition."""

import argparse
import sys
import time

import cv2
import mediapipe as mp
import pyttsx3

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

my_account = 1234567
destination_account = 0
my_balance = 10000
transact_amt = 0
transaction_state = 0

engine = pyttsx3.init()


def transact(input_gesture: str) -> None:
    global my_account, destination_account, my_balance, transact_amt, transaction_state, engine
    if transaction_state == 0 and input_gesture == 'Pointing_Up':
        # send money
        engine.say(
            text="Enter the account to send money to")
        engine.runAndWait()
        destination_account = int(input("Enter the Account to send money to: "))
        engine.say(
            text="Enter the amount to send")
        engine.runAndWait()
        transact_amt = int(input("Enter the amount to send: "))
        transaction_state = 1
        print("Show a thumbs up for confirming the transaction and thumbs down to cancel")
        engine.say(
            text="Show a thumbs up for confirming the transaction and thumbs down to cancel")
        engine.runAndWait()
    elif transaction_state == 0 and input_gesture == 'Victory':
        # demand money
        engine.say(
            text="Enter the account to ask money from")
        engine.runAndWait()
        destination_account = int(input("Enter the Account to ask money from: "))
        engine.say(
            text="Enter the amount to ask")
        engine.runAndWait()
        transact_amt = int(input("Enter the amount to ask: "))
        transaction_state = 2
        print("Show a thumbs up for confirming the transaction and thumbs down to cancel")
        engine.say(
            text="Show a thumbs up for confirming the transaction and thumbs down to cancel")
        engine.runAndWait()
    elif transaction_state == 1 and input_gesture == 'Thumb_Up':
        # authorized pay transaction
        transaction_state = 0
        my_balance = my_balance - transact_amt
        print(
            f"₹{transact_amt} paid from account {my_account} (Balance ₹{my_balance}) to {destination_account}.")
        engine.say(
            text=f"₹{transact_amt} paid from account {my_account} (Balance ₹{my_balance}) to {destination_account}.")
        engine.runAndWait()
    elif transaction_state == 2 and input_gesture == 'Thumb_Up':
        # authorized collect transaction
        transaction_state = 0
        my_balance = my_balance + transact_amt
        print(
            f"₹{transact_amt} received from account {destination_account} to {my_account}(Balance ₹{my_balance}).")
        engine.say(
            text=f"₹{transact_amt} received from account {destination_account} to {my_account}(Balance ₹{my_balance}).")
        engine.runAndWait()
    elif transaction_state == 1 or transaction_state == 2 and input_gesture == 'Thumb_Down':
        # unauthorized transaction
        transaction_state = 0
        transact_amt = 0
        print(f"Transaction Cancelled!! account {my_account} has balance ₹{my_balance}.")
        engine.say(text=f"Transaction Cancelled!! account {my_account} has balance ₹{my_balance}.")
        engine.runAndWait()
    pass


def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    global engine

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    recognition_frame = None
    recognition_result_list = []

    def save_result(result: vision.GestureRecognizerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        recognition_result_list.append(result)
        COUNTER += 1

    # Initialize the gesture recognizer model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.GestureRecognizerOptions(base_options=base_options,
                                              running_mode=vision.RunningMode.LIVE_STREAM,
                                              num_hands=num_hands,
                                              min_hand_detection_confidence=min_hand_detection_confidence,
                                              min_hand_presence_confidence=min_hand_presence_confidence,
                                              min_tracking_confidence=min_tracking_confidence,
                                              result_callback=save_result)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    fs = {"gesture": "", "count": 0}
    engine.say(text="Show a pointer for paying or victory sign for receiving money")
    engine.runAndWait()
    print("Show a pointer for paying or victory sign for receiving cancel")
    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run gesture recognizer using the model.
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
        # Show the FPS
        fps_text = 'FPS = {:.2f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if recognition_result_list:
            # Draw landmarks and write the text for each hand.
            for hand_index, hand_landmarks in enumerate(recognition_result_list[0].hand_landmarks):
                # Get gesture classification results
                if recognition_result_list[0].gestures:
                    gesture = recognition_result_list[0].gestures[hand_index]
                    category_name = gesture[0].category_name
                    score = round(gesture[0].score, 2)
                    result_text = f'{category_name} ({score})'
                    # print(category_name,fs)
                    if category_name == 'None':
                        pass
                    elif category_name == 'ILoveYou':
                        fs = {"gesture": "", "count": 0}
                        print('reset')
                    elif fs["count"] == 0 and fs["gesture"] != 'None':
                        fs = {"gesture": category_name, "count": 1}
                    elif fs["count"] == 40 and fs["gesture"] != 'None':
                        fs = {"gesture": category_name, "count": 1}
                        print(result_text)
                        transact(category_name)
                    elif fs["gesture"] == category_name:
                        fs["count"] = fs["count"] + 1

            recognition_frame = current_frame
            recognition_result_list.clear()

        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of gesture recognition model.',
        required=False,
        default='gesture_recognizer.task')
    parser.add_argument(
        '--numHands',
        help='Max number of hands that can be detected by the recognizer.',
        required=False,
        default=1)
    parser.add_argument(
        '--minHandDetectionConfidence',
        help='The minimum confidence score for hand detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minHandPresenceConfidence',
        help='The minimum confidence score of hand presence score in the hand '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the hand tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting at 0
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=480)
    args = parser.parse_args()

    run(args.model, int(args.numHands), args.minHandDetectionConfidence,
        args.minHandPresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
