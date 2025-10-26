import cv2
import argparse
import time

from ultralytics import YOLO

from main_function_webcam import camera_human_seg_view_test

##########################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', required = False, default = './models/yolov8m-seg.pt', 
                    type = str, help = 'body detection 모델 경로 입력')
parser.add_argument('--delay_time', required = False, default = 2., 
                    type = float, help = '딜레이 시간 입력')

parser.add_argument('--camera', required = False, default = 0, 
                    type = int, help = '카메라 번호')
parser.add_argument('--camera_w', required = False, default = 1280, 
                    type = int, help = '카메라 출력 너비')
parser.add_argument('--camera_h', required = False, default = 960, 
                    type = int, help = '카메라 출력 높이')
parser.add_argument('--x', required = False, default = 640, 
                    type = int, help = '타겟이 있는지 확인하고자 하는 x죄표')
parser.add_argument('--y', required = False, default = 480, 
                    type = int, help = '타겟이 있는지 확인하고자 하는 y죄표')

parser.add_argument('--threshold', required = False, default = 0.8, 
                    type = float, help = '박스 검출 문턱값')

args = parser.parse_args()

##########################################################################################

model = args.model_path

model = YOLO(model)

delay_time = args.delay_time

cap = cv2.VideoCapture(args.camera) # 노트북 웹캠을 카메라로 사용 0:노트북, 1: 웹캠
cap.set(3,args.camera_w) # 너비 640
cap.set(4,args.camera_h) # 높이 480

x = args.x
y = args.y

threshold = args.threshold

##########################################################################################

while(True):
    
    elapsed_time = camera_human_seg_view_test(model, cap, x, y, threshold)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

    sleep_time = max(delay_time - elapsed_time, 0)
    time.sleep(sleep_time)
    
##########################################################################################
