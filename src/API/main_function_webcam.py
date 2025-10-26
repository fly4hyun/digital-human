import cv2
import numpy as np
import time

##########################################################################################

# 박스 설정
blue_color_100 = (255, 200, 200)
g_color_100 = (200, 255, 200)
red_color_100 = (200, 200, 255)
blue_color = (255, 0, 0)
g_color = (0, 255, 0)
red_color = (0, 0, 255)
center_color = (150, 150, 150)
thickness = 2
keypoint_thickness = 10
box_threshold = 0.800

##########################################################################################

def LiDAR_human_measure(pipeline, 
                        align, 
                        x_y, 
                        ):
    
    start_time = time.time()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    depth_info = depth_frame.as_depth_frame()
    
    measures = []
    for x, y in x_y:
        measures.append(round((depth_info.get_distance(x, y) * 100), 2))
        
    measure = 0
    num = 0
    for mea in measures:
        if mea != 0:
            num += 1
            measure += mea
    if num == 0:
        num = 1
    measure = measure / num

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return measure, elapsed_time

##########################################################################################

def LiDAR_human_measure_view(pipeline, 
                             align, 
                             x_y
                             ):
    
    start_time = time.time()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_info = depth_frame.as_depth_frame()
    
    measures = []
    for x, y in x_y:
        measures.append(round((depth_info.get_distance(x, y) * 100), 2))
        
    measure = 0
    num = 0
    for mea in measures:
        if mea != 0:
            num += 1
            measure += mea
    if num == 0:
        num = 1
    measure = measure / num
    
    
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    i = 0
    for x, y in x_y:
        color_image = cv2.circle(color_image, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(color_image, str(measures[i])[:3], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness)
        i+=1
    cv2.imshow('RealSense', color_image)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return measure, elapsed_time

##########################################################################################

def camera_human_detection(body_model, 
                    cap, 
                    x, 
                    y, 
                    box_threshold):
    
    start_time = time.time()
    check_human = 0
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭

    body_results = body_model(frame, verbose = False)[0]

    body_boxes_info = body_results.boxes
    body_boxes = body_boxes_info.xyxy.tolist()
    body_boxes_conf = body_boxes_info.conf.tolist()
    
    body_keypoints = body_results.keypoints.xy.tolist()
    
    #print("Number of faces detected: " + str(len(body_boxes)))
    
    for box_index in range(len(body_boxes)):
        
        body_box = body_boxes[box_index]
        if body_boxes_conf[box_index] < box_threshold:
            continue
        
        body_box = list(map(int, body_box))
        
        body_box_arr = np.array([body_box[:2], 
                                 [body_box[0], body_box[3]], 
                                 body_box[2:], 
                                 [body_box[2], body_box[1]]])

        center_box_result = cv2.pointPolygonTest(body_box_arr, (x, y), False)
        if center_box_result == -1:
            continue
        
        body_keypoint_1 = np.array([list(map(int, body_keypoints[box_index][5])), 
                                    list(map(int, body_keypoints[box_index][6])), 
                                    list(map(int, body_keypoints[box_index][12])), 
                                    list(map(int, body_keypoints[box_index][11]))])
        
        body_keypoint_2 = np.array([list(map(int, body_keypoints[box_index][5])), 
                                    list(map(int, body_keypoints[box_index][6])), 
                                    list(map(int, body_keypoints[box_index][8])), 
                                    list(map(int, body_keypoints[box_index][7]))])
        
        center_keypoint_result_1 = cv2.pointPolygonTest(body_keypoint_1, (x, y), False)
        center_keypoint_result_2 = cv2.pointPolygonTest(body_keypoint_2, (x, y), False)
        if center_keypoint_result_1 == -1 and center_keypoint_result_2 == -1:
            continue
        
        check_human = 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return check_human, elapsed_time

##########################################################################################

def camera_human_detection_view(body_model, 
                         face_model, 
                         cap, 
                         x, 
                         y, 
                         box_threshold):
    
    start_time = time.time()
    check_human = 0
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭

    body_results = body_model(frame, verbose = False)[0]
    face_boxes = face_model(frame, verbose = False)[0].boxes.xyxy.tolist()

    body_boxes_info = body_results.boxes
    body_boxes = body_boxes_info.xyxy.tolist()
    body_boxes_conf = body_boxes_info.conf.tolist()
    
    body_keypoints = body_results.keypoints.xy.tolist()
    
    for box_index in range(len(body_boxes)):
        
        body_box = body_boxes[box_index]
        if body_boxes_conf[box_index] < box_threshold:
            continue
        
        body_box = list(map(int, body_box))
        
        body_box_arr = np.array([body_box[:2], 
                                 [body_box[0], body_box[3]], 
                                 body_box[2:], 
                                 [body_box[2], body_box[1]]])

        center_box_result = cv2.pointPolygonTest(body_box_arr, (x, y), False)
        if center_box_result == -1:
            continue
        
        body_keypoint_1 = np.array([list(map(int, body_keypoints[box_index][5])), 
                                    list(map(int, body_keypoints[box_index][6])), 
                                    list(map(int, body_keypoints[box_index][12])), 
                                    list(map(int, body_keypoints[box_index][11]))])
        
        body_keypoint_2 = np.array([list(map(int, body_keypoints[box_index][5])), 
                                    list(map(int, body_keypoints[box_index][6])), 
                                    list(map(int, body_keypoints[box_index][8])), 
                                    list(map(int, body_keypoints[box_index][7]))])
        
        center_keypoint_result_1 = cv2.pointPolygonTest(body_keypoint_1, (x, y), False)
        center_keypoint_result_2 = cv2.pointPolygonTest(body_keypoint_2, (x, y), False)
        if center_keypoint_result_1 == -1 and center_keypoint_result_2 == -1:
            continue
        
        check_human = 1
        
        x_noes = int(body_keypoints[box_index][0][0])
        y_noes = int(body_keypoints[box_index][0][1])

        for face_box in face_boxes:
    
            face_box = list(map(int, face_box))
            
            face_box_arr = np.array([face_box[:2], 
                            [face_box[0], face_box[3]], 
                            face_box[2:], 
                            [face_box[2], face_box[1]]])
            
            face_box_result = cv2.pointPolygonTest(face_box_arr, (x_noes, y_noes), False)
            
            if face_box_result == -1:
                continue
            
            frame = cv2.rectangle(frame, face_box[:2], face_box[2:], red_color, thickness)
        
        body_box = list(map(int, body_box))
        frame = cv2.rectangle(frame, body_box[:2], body_box[2:], blue_color, thickness)
        
        body_keypoint_1 = body_keypoint_1.reshape((-1, 1, 2))
        frame = cv2.polylines(frame, [body_keypoint_1], isClosed = True, color = g_color, thickness = thickness)
        body_keypoint_2 = body_keypoint_2.reshape((-1, 1, 2))
        frame = cv2.polylines(frame, [body_keypoint_2], isClosed = True, color = g_color, thickness = thickness)
        
    frame = cv2.line(frame, [x, y], [x, y], center_color, keypoint_thickness)
    
    ##########################################################################################
    
    cv2.imshow('result', frame)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return check_human, elapsed_time

##########################################################################################

def camera_human_detection_view_test(body_model, 
                              face_model, 
                              cap, 
                              x, 
                              y, 
                              box_threshold):
    
    start_time = time.time()
    check_human = 0
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭

    body_results = body_model(frame, verbose = False)[0]
    face_boxes = face_model(frame, verbose = False)[0].boxes.xyxy.tolist()

    body_boxes_info = body_results.boxes
    body_boxes = body_boxes_info.xyxy.tolist()
    body_boxes_conf = body_boxes_info.conf.tolist()
    
    body_keypoints = body_results.keypoints.xy.tolist()
    
    #########################################################################################
    #### 검출되는 모든 사물들 출력
    
    for box_index in range(len(body_boxes)):
        
        body_box = body_boxes[box_index]
        if body_boxes_conf[box_index] < box_threshold:
            continue
        
        body_box = list(map(int, body_box))
        frame = cv2.rectangle(frame, body_box[:2], body_box[2:], blue_color_100, thickness)
        
        for point in body_keypoints[box_index]:
            if [0, 0] == point:
                continue
            
            x_point, y_point = point
            x_point = int(x_point)
            y_point = int(y_point)
            
            frame = cv2.line(frame, [x_point, y_point], [x_point, y_point], g_color_100, keypoint_thickness)
    
    for face_box in face_boxes:
        
        face_box = list(map(int, face_box))
        frame = cv2.rectangle(frame, face_box[:2], face_box[2:], red_color_100, thickness)
    
    frame = cv2.line(frame, [x, y], [x, y], g_color, keypoint_thickness)
    
    #########################################################################################
    
    for box_index in range(len(body_boxes)):
        
        body_box = body_boxes[box_index]
        if body_boxes_conf[box_index] < box_threshold:
            continue
        
        body_box = list(map(int, body_box))
        
        body_box_arr = np.array([body_box[:2], 
                                 [body_box[0], body_box[3]], 
                                 body_box[2:], 
                                 [body_box[2], body_box[1]]])

        center_box_result = cv2.pointPolygonTest(body_box_arr, (x, y), False)
        if center_box_result == -1:
            continue
        
        body_keypoint_1 = np.array([list(map(int, body_keypoints[box_index][5])), 
                                    list(map(int, body_keypoints[box_index][6])), 
                                    list(map(int, body_keypoints[box_index][12])), 
                                    list(map(int, body_keypoints[box_index][11]))])
        
        body_keypoint_2 = np.array([list(map(int, body_keypoints[box_index][5])), 
                                    list(map(int, body_keypoints[box_index][6])), 
                                    list(map(int, body_keypoints[box_index][8])), 
                                    list(map(int, body_keypoints[box_index][7]))])
        
        center_keypoint_result_1 = cv2.pointPolygonTest(body_keypoint_1, (x, y), False)
        center_keypoint_result_2 = cv2.pointPolygonTest(body_keypoint_2, (x, y), False)
        if center_keypoint_result_1 == -1 and center_keypoint_result_2 == -1:
            continue
        
        check_human = 1
        
        x_noes = int(body_keypoints[box_index][0][0])
        y_noes = int(body_keypoints[box_index][0][1])

        for face_box in face_boxes:
    
            face_box = list(map(int, face_box))
            
            face_box_arr = np.array([face_box[:2], 
                            [face_box[0], face_box[3]], 
                            face_box[2:], 
                            [face_box[2], face_box[1]]])
            
            face_box_result = cv2.pointPolygonTest(face_box_arr, (x_noes, y_noes), False)
            
            if face_box_result == -1:
                continue
            
            frame = cv2.rectangle(frame, face_box[:2], face_box[2:], red_color, thickness)
        
        body_box = list(map(int, body_box))
        frame = cv2.rectangle(frame, body_box[:2], body_box[2:], blue_color, thickness)
        
        body_keypoint_1 = body_keypoint_1.reshape((-1, 1, 2))
        frame = cv2.polylines(frame, [body_keypoint_1], isClosed = True, color = g_color, thickness = thickness)
        body_keypoint_2 = body_keypoint_2.reshape((-1, 1, 2))
        frame = cv2.polylines(frame, [body_keypoint_2], isClosed = True, color = g_color, thickness = thickness)
        
    frame = cv2.line(frame, [x, y], [x, y], center_color, keypoint_thickness)
    
    ##########################################################################################
    
    cv2.imshow('result', frame)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return check_human, elapsed_time

##########################################################################################

