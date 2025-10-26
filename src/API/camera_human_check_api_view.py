from flask import Flask, request, jsonify
import threading
from main_function import camera_human_detection_view_test
from ultralytics import YOLO
import cv2
import numpy as np
import time

app = Flask(__name__)

detection_status = '0'
elapsed_time = 0

def camera_human_detection_func():
    global detection_status
    global elapsed_time
    while True:
        # 사람이 있는지 없는지 랜덤하게 '1' 또는 '0'을 설정
        detection_status, elapsed_time = camera_human_detection_view_test(body_model, face_model, cap, x, y, threshold)
        detection_status = str(detection_status)

@app.route('/detect-human', methods=['POST'])
def detect_human():
    
    return jsonify({'human_detected': detection_status})

if __name__ == '__main__':
    
    body_path = './models/yolov8n-pose.pt'
    face_path = './models/yolov8n-face.pt'
    
    body_model = YOLO(body_path)
    face_model = YOLO(face_path)
    
    cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용 0:노트북, 1: 웹캠
    cap.set(3,1280) # 너비 640
    cap.set(4,960) # 높이 480
    
    delay_time = 1
    
    # x, y 좌표와 임계값에 대한 임시 값
    x = 640
    y = 480
    threshold = 0.8
    
    threading.Thread(target=camera_human_detection_func, daemon=True).start()
    
    sleep_time = max(delay_time - elapsed_time, 0)
    time.sleep(sleep_time)
    
    app.run(debug=False)
