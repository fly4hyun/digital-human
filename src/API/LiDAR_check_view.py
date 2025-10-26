import cv2
import argparse
import pyrealsense2 as rs
import numpy as np
import time

from main_function import LiDAR_human_measure_view

##########################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--delay_time', required = False, default = 1., 
                    type = float, help = '딜레이 시간 입력')
parser.add_argument('--LiDAR_w', required = False, default = 640, 
                    type = int, help = '라이다 너비 정보')
parser.add_argument('--LiDAR_h', required = False, default = 480, 
                    type = int, help = '라이다 높이 정보')
parser.add_argument('--x', required = False, default = 320, 
                    type = int, help = '타겟이 있는지 확인하고자 하는 x죄표')
parser.add_argument('--y', required = False, default = 240, 
                    type = int, help = '타겟이 있는지 확인하고자 하는 y죄표')

args = parser.parse_args()

##########################################################################################

delay_time = args.delay_time

x = args.x
y = args.y

x_y = [[x-10, y-10], [x-10, y+10], [x+10, y-10], [x+10, y+10], [x, y], 
       [x-15, y], [x+15, y], [x, y-15], [x, y+15]]

##########################################################################################

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

config.enable_stream(rs.stream.depth, args.LiDAR_w, args.LiDAR_h, rs.format.z16, 30)
config.enable_stream(rs.stream.color, args.LiDAR_w, args.LiDAR_h, rs.format.bgr8, 30)

##########################################################################################

pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

while True:

    measure, elapsed_time = LiDAR_human_measure_view(pipeline, align, x_y)
    
    print(measure)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break
    
    sleep_time = max(delay_time - elapsed_time, 0)
    time.sleep(sleep_time)