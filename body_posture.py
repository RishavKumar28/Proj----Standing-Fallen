import cv2
import mediapipe as mp
import math
import pygame
import requests
import os
import smtplib
import time
import threading
import urllib.parse
from email.message import EmailMessage
from datetime import datetime
from ultralytics import YOLO
import geocoder

# === Email Alert Configuration ===
EMAIL_SENDER = 'abc@gmail.com'
EMAIL_PASSWORD = '*** *** *** **'
EMAIL_RECEIVER = 'def@gmail.com'
HARDCODED_ADDRESS = " #######################################"

def send_email_alert(location, timestamp, video_path, map_link=""):
    try:
        msg = EmailMessage()
        msg.set_content(f"""🚨 Alert: A fall has been detected.

Timestamp: {timestamp}
Address: {location}

Google Maps: {map_link if map_link else 'Unavailable'}

Please check the attached video for details.
""")
        msg['Subject'] = "🚨 Fall Detected Alert!"
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        with open(video_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(video_path)
            msg.add_attachment(file_data, maintype='video', subtype='x-msvideo', filename=file_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("📨 Email with video sent.")
    except Exception as e:
        print("❌ Failed to send email:", e)

def get_location():
    try:
        encoded_address = urllib.parse.quote(HARDCODED_ADDRESS)
        map_link = f"https://www.google.com/maps/search/?api=1&query={encoded_address}"

        with open("location_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} - {HARDCODED_ADDRESS} - Map: {map_link}\n")

        return HARDCODED_ADDRESS, map_link
    except Exception as e:
        print("Location error:", e)
        return HARDCODED_ADDRESS, "Map link unavailable"

pygame.mixer.init()
ALARM_FILE = r"$$$$$"
pygame.mixer.music.load(ALARM_FILE)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p3):
    a = [p1.x - p2.x, p1.y - p2.y]
    b = [p3.x - p2.x, p3.y - p2.y]
    dot = a[0]*b[0] + a[1]*b[1]
    mag_a = math.sqrt(a[0]**2 + a[1]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2)
    if mag_a * mag_b == 0:
        return 0
    angle = math.acos(dot / (mag_a * mag_b))
    return math.degrees(angle)

def classify_posture(landmarks, prev_shoulder):
    try:
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        angle = calculate_angle(shoulder, hip, knee)

        if prev_shoulder:
            dx = shoulder.x - prev_shoulder.x
            dy = shoulder.y - prev_shoulder.y
            movement = math.sqrt(dx**2 + dy**2)
            if movement > 0.03:
                return "Moving"

        return "Standing" if angle > 170 else "Fallen"
    except:
        return "Unknown"

os.makedirs("recordings", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"recordings/fall_recording_{timestamp}.avi"

model = YOLO("yolov8n.pt")
model.fuse()
model.overrides['conf'] = 0.6

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 10.7, (frame_width, frame_height))

alarm_playing = False
location_display = ""
timestamp_display = ""
prev_shoulder_dict = {}
email_sent_global = 0
MAX_EMAIL_LIMIT = 2

FRAME_SKIP = 6
frame_counter = 0
last_people_boxes = []

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame_counter += 1
    if frame_counter % FRAME_SKIP == 0:
        people_boxes = []
        results = model(frame)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                people_boxes.append((x1, y1, x2, y2))
        last_people_boxes = people_boxes
    else:
        people_boxes = last_people_boxes

    person_count = len(people_boxes)
    fallen_detected = False

    for i, (x1, y1, x2, y2) in enumerate(people_boxes):
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue

        rgb_crop = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb_crop)

        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_result.pose_landmarks.landmark
            prev_shoulder = prev_shoulder_dict.get(i)
            posture = classify_posture(landmarks, prev_shoulder)
            prev_shoulder_dict[i] = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            if posture == "Standing":
                color = (0, 255, 0)
            elif posture == "Fallen":
                color = (0, 0, 255)
            elif posture == "Moving":
                color = (255, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.putText(frame, f"Person {i+1}: {posture}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if posture == "Fallen":
                fallen_detected = True

    if fallen_detected:
        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True

        if email_sent_global < MAX_EMAIL_LIMIT:
            def send_async_alert():
                global location_display, timestamp_display
                location, map_link = get_location()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                location_display = location
                timestamp_display = timestamp
                send_email_alert(location, timestamp, video_path, map_link)

            threading.Thread(target=send_async_alert, daemon=True).start()
            email_sent_global += 1
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False
        location_display = ""
        timestamp_display = ""

    cv2.putText(frame, f"People Detected: {person_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if fallen_detected:
        cv2.putText(frame, f"Location:", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"{HARDCODED_ADDRESS}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Time: {timestamp_display}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(frame)
    cv2.imshow("Multi-Person Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()

print(f"📁 Recording saved as: {video_path}")
