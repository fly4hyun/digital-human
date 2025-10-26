import cv2
import argparse
import pyrealsense2 as rs
import numpy as np
import time

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

    start_time = time.time()
    
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_info = depth_frame.as_depth_frame()

    print("Depth : ", round((depth_info.get_distance(args.x, args.y) * 100), 2), "cm")

    color_image = np.asanyarray(color_frame.get_data())
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    color_image = cv2.circle(color_image, (args.x, args.y), 2, (0, 0, 255), -1)
    color_image = cv2.flip(color_image, 1)
    cv2.imshow('RealSense', color_image)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

    sleep_time = max(args.delay_time - elapsed_time, 0)
    time.sleep(sleep_time)