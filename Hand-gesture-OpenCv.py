import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the pre-trained model
newmod = load_model('hand_gestures_1.h5')

# Background model for averaging
background = None
accumulated_weight = 0.5

# Region of Interest (ROI) coordinates
roi_top = 50
roi_bottom = 350
roi_right = 400
roi_left = 700

# Configuration constants
update_bg_frames = 60
prediction_delay = 5

# Calculate accumulated average for background subtraction
def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

# Segment the hand from the background
def segment(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    thresholded = cv2.erode(thresholded, None, iterations=2)
    thresholded = cv2.dilate(thresholded, None, iterations=2)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)

# Predict gesture using the model
def thres_display(img):
    width, height = 64, 64
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    test_img = image.img_to_array(resized)
    test_img = np.expand_dims(test_img, axis=0)
    result = newmod.predict(test_img)
    val = np.argmax(result)  # Find the class with the highest probability
    return val

# Countdown timer for initialization
def countdown(seconds=5):
    for i in range(seconds, 0, -1):
        print(f"Starting in {i} seconds...")
        time.sleep(1)
    print("Go!")

# Start countdown before initializing the camera
countdown()

# Initialize webcam
cam = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Define the Region of Interest (ROI)
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Update the background model
    if num_frames < update_bg_frames:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= update_bg_frames - 1:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Display instructions
        cv2.putText(frame_copy, "Place your hand inside the box", (330, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 0: Fist", (330, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 1: Five", (330, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 3: Okay", (330, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 4: Peace", (330, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 5: Rad", (330, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 6: Straight", (330, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame_copy, "Index 7: Thumbs", (330, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Segment the hand from the background
    hand = segment(gray)
    if hand is not None:
        thresholded, hand_segment = hand
        cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
        cv2.imshow("Thresholded Image", thresholded)

        # Display the predicted gesture
        res = thres_display(thresholded)
        gesture = f"Index {res}"
        cv2.putText(frame_copy, gesture, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the ROI rectangle
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)
    num_frames += 1
    cv2.imshow("Hand Gestures", frame_copy)

    # Exit loop on ESC key press
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
