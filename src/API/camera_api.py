from flask import Flask, request, jsonify
import threading
from main_function import camera_human_detection
from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np

app = Flask(__name__)

detection_status = '0'
elapsed_time = 0

def camera_human_detection_func():
    global detection_status
    global elapsed_time
    while True:
        # 사람이 있는지 없는지 랜덤하게 '1' 또는 '0'을 설정
        detection_status, elapsed_time = camera_human_detection(body_model, pipeline, align, x, y, threshold)
        detection_status = str(detection_status)

@app.route('/detect-human', methods=['POST'])
def detect_human():
    
    return jsonify({'human_detected': detection_status, 'elapsed_time': elapsed_time})

if __name__ == '__main__':
    
    body_path = './models/yolov8n-pose.pt'
    face_path = './models/yolov8n-face.pt'
    
    body_model = YOLO(body_path)
    face_model = YOLO(face_path)
    
    # cap = cv2.VideoCapture(args.camera) # 노트북 웹캠을 카메라로 사용 0:노트북, 1: 웹캠
    # cap.set(3,args.camera_w) # 너비 640
    # cap.set(4,args.camera_h) # 높이 480

    #############

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break

    #config.enable_stream(rs.stream.depth, args.LiDAR_w, args.LiDAR_h, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    #############
    
    delay_time = 1
    
    # x, y 좌표와 임계값에 대한 임시 값
    x = 320
    y = 240
    threshold = 0.8
    
    threading.Thread(target=camera_human_detection_func, daemon=True).start()
    app.run(debug=False, port=5000)
