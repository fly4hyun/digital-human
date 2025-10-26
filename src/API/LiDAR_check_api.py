from flask import Flask, request, jsonify
import threading

import argparse
import pyrealsense2 as rs
import time

from main_function import LiDAR_human_measure

##########################################################################################

app = Flask(__name__)

measure_status = 0
elapsed_time = 0

def LiDAR_human_measure_func():
    global measure_status
    global elapsed_time
    while True:
        # 사람이 있는지 없는지 랜덤하게 '1' 또는 '0'을 설정
        measure_status, elapsed_time = LiDAR_human_measure(pipeline, align, x_y)
        #measure_status = str(measure_status)

@app.route('/measure-human', methods=['POST'])
def detect_human():
    
    return jsonify({'human_measured': measure_status, 'elapsed_time': elapsed_time})


##########################################################################################

if __name__ == '__main__':
    
    delay_time = 1
    
    x = 320
    y = 240
    
    x_y = [[x-10, y-10], [x-10, y+10], [x+10, y-10], [x+10, y+10], [x, y], 
       [x-15, y], [x+15, y], [x, y-15], [x, y+15]]
    
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    threading.Thread(target=LiDAR_human_measure_func, daemon=True).start()
    app.run(debug=False, port=5001)

