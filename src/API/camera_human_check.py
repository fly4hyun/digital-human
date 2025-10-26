import pyrealsense2 as rs
import cv2
import argparse
import time

from ultralytics import YOLO

from main_function import camera_human_detection

##########################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--body_model_path', required = False, default = './models/yolov8n-pose.pt', 
                    type = str, help = 'body detection 모델 경로 입력')
parser.add_argument('--face_model_path', required = False, default = './models/yolov8n-face.pt', 
                    type = str, help = 'face detection 모델 경로 입력')
parser.add_argument('--delay_time', required = False, default = 1., 
                    type = float, help = '딜레이 시간 입력')

parser.add_argument('--camera', required = False, default = 0, 
                    type = int, help = '카메라 번호')
parser.add_argument('--camera_w', required = False, default = 640, 
                    type = int, help = '카메라 출력 너비')
parser.add_argument('--camera_h', required = False, default = 480, 
                    type = int, help = '카메라 출력 높이')
parser.add_argument('--x', required = False, default = 320, 
                    type = int, help = '타겟이 있는지 확인하고자 하는 x죄표')
parser.add_argument('--y', required = False, default = 240, 
                    type = int, help = '타겟이 있는지 확인하고자 하는 y죄표')

parser.add_argument('--threshold', required = False, default = 0.8, 
                    type = float, help = '박스 검출 문턱값')

args = parser.parse_args()

##########################################################################################

body_path = args.body_model_path
face_path = args.face_model_path

body_model = YOLO(body_path)
face_model = YOLO(face_path)

delay_time = args.delay_time

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
config.enable_stream(rs.stream.color, args.camera_w, args.camera_h, rs.format.bgr8, 30)

pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

#############

x = args.x
y = args.y

threshold = args.threshold

##########################################################################################

while(True):
    
    check, elapsed_time = camera_human_detection(body_model, pipeline, align, x, y, threshold)
    
    print(check)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break
    
    sleep_time = max(delay_time - elapsed_time, 0)
    time.sleep(sleep_time)

##########################################################################################
