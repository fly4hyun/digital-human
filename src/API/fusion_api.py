from flask import Flask, request, jsonify
import threading
import pyrealsense2 as rs
import cv2
import argparse
import time

from ultralytics import YOLO

from main_function import LiDAR_camera_function

##########################################################################################

app = Flask(__name__)

measure_status = 0
detection_status = '0'
human_status = '0'
elapsed_time = 0

##########################################################################################

def LiDAR_human_measure_func():
    global measure_status
    global detection_status
    global human_status
    global elapsed_time
    while True:
        # 사람이 있는지 없는지 랜덤하게 '1' 또는 '0'을 설정합니다.
        measure_status, detection_status, elapsed_time = LiDAR_camera_function(pipeline, align, body_model, face_model, x, y, x_y, threshold)
        detection_status = str(detection_status)
        
        if measure_status < distance_human and measure_status > distance_human - 40 and detection_status == '1':
            human_status = '1'

##########################################################################################

@app.route('/check-human', methods=['POST'])
def check_human():

    return jsonify({'human': human_status,'human_measured': measure_status, 'human_detected': detection_status, 'elapsed_time': elapsed_time})

##########################################################################################

if __name__ == '__main__':

    ########## 카메라 사람 탐지 세팅 ##########
    
    body_path = './models/yolov8m-pose.pt'
    face_path = './models/yolov8n-face.pt'
    
    body_model = YOLO(body_path)
    face_model = YOLO(face_path)
    
    delay_time = 1
    threshold = 0.8
    
    ########## 라이다 거리 측정 세팅 ##########
    
    x = 320
    y = 240
    
    x_y = [[x-10, y-10], [x-10, y+10], [x+10, y-10], [x+10, y+10], [x, y], 
       [x-15, y], [x+15, y], [x, y-15], [x, y+15]]
    
    distance_human = 100
    
    ########## 리얼센스 세팅 ##########

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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    threading.Thread(target=LiDAR_human_measure_func, daemon=True).start()
    app.run(debug=False, port=5000)


