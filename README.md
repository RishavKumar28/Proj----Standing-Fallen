# Proj----Standing-Fallen
AI-powered multi-person fall detection system using YOLOv8, MediaPipe, and OpenCV. Detects standing/moving/fallen posture in real time, triggers alarm, records video, and sends email alerts with location and timestamp. Ideal for elderly care, hospitals, and safety monitoring.

--- For libraries---
 
LIBRARY INSTALLATION GUIDE

This document lists the required Python packages and provides the command to install them.

The libraries listed below are used for the following functions in the system:

opencv-python: Core library for handling video streams, camera input, and frame manipulation (used as cv2).

mediapipe: Used for high-fidelity human pose estimation and tracking.

ultralytics: Used for running the YOLOv8 model for initial person detection.

pygame: Used for playing the local audio alarm when a fall is detected.

requests: General-purpose HTTP library for making web requests (e.g., to geocoding services).

geocoder / geopy: Used together for finding the public IP location and converting coordinates to a readable address.

python-dotenv: Used for securely loading the EMAIL_PASSWORD from the .env file.

smtplib: Standard Python library for sending email notifications.

math, os, time, threading, urllib.parse, email.message, datetime: Standard Python libraries that are included with Python and do not require separate installation.

INSTALLATION COMMAND

To install all necessary third-party libraries, use the following single command in your terminal or command prompt:

pip install opencv-python mediapipe pygame requests geocoder geopy python-dotenv ultralytics
