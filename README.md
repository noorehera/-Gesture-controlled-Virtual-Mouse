# Gesture-Controlled Virtual Mouse

This project enables a **gesture-controlled virtual mouse** using the webcam, hand gestures, and Python libraries such as **OpenCV**, **MediaPipe**, and **PyAutoGUI**. The user can control the mouse pointer and click using hand gestures detected by the webcam.

## Features

1. **Mouse Movement**: Track the movement of your hand (index finger) to control the mouse cursor.
2. **Mouse Click**: Perform a mouse click when your thumb and index finger are close together (pinch gesture).
3. **Smooth Hand Tracking**: Uses MediaPipe to accurately detect hand landmarks and track hand movement.
4. **Cross-platform Support**: Works on Windows, macOS, and Linux systems that support Python and the necessary libraries.

## Prerequisites

Before running the program, make sure you have the following installed:

- **Python 3.x**: The programming language used for this project.
- **Libraries**: You will need to install a few Python libraries, including:
  - **OpenCV** for video capture and image processing.
  - **MediaPipe** for hand tracking.
  - **PyAutoGUI** for controlling the mouse.

To install the required libraries, run the following commands in your terminal:

```bash
pip install opencv-python mediapipe pyautogui
