import cv2
import numpy as np
import torch
import pyttsx3
from ultralytics import YOLO
from collections import deque

# ====== INIT MODEL AND TTS ======
model = YOLO('best_drone_model.pt')  # Your trained YOLOv8 model
tts = pyttsx3.init()
tts.setProperty('rate', 145)

def speak(msg):
    print(f"[SPEAK] {msg}")
    tts.say(msg)
    tts.runAndWait()

# ====== FEATURE 1: EMERGENCY VEHICLE FLASHING LIGHT DETECTION ======
class EmergencyFlashingLightDetector:
    def __init__(self):
        self.buffer = deque(maxlen=10)
        self.alarmed = False

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
        blue = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)
        r = cv2.countNonZero(red_mask)
        b = cv2.countNonZero(blue)
        self.buffer.append(r > 300 and b > 300)
        if all(self.buffer) and not self.alarmed:
            speak("Emergency vehicle detected. Please clear the way.")
            self.alarmed = True
        if not any(self.buffer):
            self.alarmed = False

# ====== FEATURE 2: LANE VIOLATION DETECTION ======
def check_lane_violation(frame, boxes, lane_y=300):
    cv2.line(frame, (0, lane_y), (frame.shape[1], lane_y), (0, 255, 255), 2)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        center_y = (y1 + y2) // 2
        if center_y < lane_y:
            cv2.putText(frame, "LANE VIOLATION", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            speak("Vehicle crossed yellow lane. Violation detected.")

# ====== FEATURE 3: WEATHER-ADAPTIVE VISION ENHANCEMENT ======
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ====== MAIN DRIVER FUNCTION ======
def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    emergency = EmergencyFlashingLightDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = enhance_frame(frame)  # Weather enhancement
        results = model(frame)[0]
        detections = results.boxes.xyxy.cpu().numpy()

        # Draw all boxes
        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        boxes = [box[:4] for box in detections]

        check_lane_violation(frame, boxes)
        emergency.detect(frame)

        cv2.imshow("Smart Drone Traffic Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====== RUN ======
if __name__ == "__main__":
    run_detection("C:/Users/iampr/Desktop/Drone intern/datasets/traffic drone shot of a city.mp4")

