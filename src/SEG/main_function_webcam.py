import cv2
import numpy as np
import time

from utils import (remove_green_screen, 
                   crop_transparent_area, 
                   blur_non_transparent_area, 
                   darken_image, 
                   soften_edges, 
                   combine_images, 
                   apply_sepia_effect, 
                   apply_film_damage_effect, 
                   apply_edge_damage_effect)

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

def camera_human_seg_view_test(model, 
                              cap, 
                              x,
                              y, 
                              box_threshold):
    
    start_time = time.time()
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭

    # 모델을 통한 마스크 감지
    model_output = model(frame, iou=box_threshold, classes=0, verbose=False)
    if not model_output or not model_output[0].masks:
        cv2.imshow('result', np.full_like(frame, [0, 255, 0], dtype=np.uint8))
        return 0  # 마스크 감지 실패 시 함수 종료

    mask_results = model_output[0].masks.data.cpu()
    
    target_mask_found = False
    processed_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # 초기화

    for i in range(mask_results.shape[0]):
        mask = mask_results[i]
        mask = np.array(mask)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # 마스크 확장
        kernel = np.ones((1, 1), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 확장된 마스크의 내부 영역 채우기
        _, binary_mask = cv2.threshold(dilated_mask, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 각 영역(컨투어)에 대해 x, y 좌표가 포함되는지 확인
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                cv2.drawContours(processed_mask, [contour], -1, 255, thickness=cv2.FILLED)
                target_mask_found = True
                break
        
        if target_mask_found:
            break

    # 마스크가 감지되지 않은 경우 초록색 배경으로 설정
    if target_mask_found:
        # 가우시안 블러로 경계 부드럽게 처리
        #blurred_mask = cv2.GaussianBlur(processed_mask, (1, 1), 0)
        #alpha_mask = blurred_mask / 255.0  # 정규화

        # 원본 frame과 초록색 배경을 픽셀별로 블렌딩
        frame_masked = cv2.bitwise_and(frame, frame, mask=processed_mask)
        background = np.full_like(frame, [0, 255, 0], dtype=np.uint8)
        background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(processed_mask))
        processed_mask = processed_mask / 255.0
        final_frame = (frame_masked * processed_mask[..., None] + background_masked * (1 - processed_mask[..., None])).astype(np.uint8)
    else:
        final_frame = np.full_like(frame, [0, 255, 0], dtype=np.uint8)
        
    ##########################################################################################
    
    cv2.imshow('result', final_frame)

    ##########################################################################################

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time

##########################################################################################

def chroma_key_func_v2(frame, pic):
    # pic 이미지의 크기를 얻기
    pic_height, pic_width = pic.shape[:2]

    # 원본 frame의 크기와 비율 계산
    frame_height, frame_width = frame.shape[:2]
    ratio_height = pic_height / frame_height
    ratio_width = pic_width / frame_width
    scale = min(ratio_height, ratio_width)

    # frame 크기를 비율에 맞게 조정
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 초록색 배경 이미지 생성
    green_background = np.full((pic_height, pic_width, 3), [0, 255, 0], dtype=np.uint8)

    # frame을 초록색 배경의 우측 하단에 배치
    x_offset = pic_width - new_width
    y_offset = pic_height - new_height
    green_background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    # 유연한 크로마키 처리를 위한 색상 범위 설정
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([0, 255, 0])
    mask = cv2.inRange(green_background, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # 마스크의 경계 감지
    edges = cv2.Canny(mask, 50, 150)

    # 경계 부분에 덜 강한 블러 처리
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))  # 더 작은 커널로 확장
    blurred_edges = cv2.GaussianBlur(dilated_edges, (5, 5), 0)  # 더 작은 커널로 블러

    # 블러 처리된 경계를 원래 마스크와 결합
    soft_mask = cv2.max(mask, blurred_edges)

    # 마스크 반전
    mask_inv = cv2.bitwise_not(soft_mask)

    # 초록색 영역을 pic 이미지로 크로마키 처리
    pic_bg = cv2.bitwise_and(pic, pic, mask=soft_mask)
    frame_fg = cv2.bitwise_and(green_background, green_background, mask=mask_inv)
    combined = cv2.add(pic_bg, frame_fg)

    return combined

##########################################################################################

def chroma_key_func(frame, pic):
    # pic 이미지의 크기를 얻기
    pic_height, pic_width = pic.shape[:2]

    # 원본 frame의 크기와 비율 계산
    frame_height, frame_width = frame.shape[:2]
    ratio_height = pic_height / frame_height
    ratio_width = pic_width / frame_width
    scale = min(ratio_height, ratio_width)

    # frame 크기를 비율에 맞게 조정
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 초록색 배경 이미지 생성
    green_background = np.full((pic_height, pic_width, 3), [0, 255, 0], dtype=np.uint8)

    # frame을 초록색 배경의 우측 하단에 배치
    x_offset = pic_width - new_width
    y_offset = pic_height - new_height
    green_background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    # 유연한 크로마키 처리를 위한 색상 범위 설정
    lower_green = np.array([0, 1, 0])
    upper_green = np.array([0, 255, 0])
    mask = cv2.inRange(green_background, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # 초록색 영역을 pic 이미지로 크로마키 처리
    pic_bg = cv2.bitwise_and(pic, pic, mask=mask)
    frame_fg = cv2.bitwise_and(green_background, green_background, mask=mask_inv)
    combined = cv2.add(pic_bg, frame_fg)

    return combined

##########################################################################################

def salt_and_pepper_mask(shape, salt_pepper_ratio=0.01):
    mask = np.random.choice([0, 1], size=shape, p=[salt_pepper_ratio, 1-salt_pepper_ratio])
    return mask.astype(np.float32)

##########################################################################################

def camera_human_seg_with_pic(model, 
                              cap, 
                              x,
                              y, 
                              box_threshold, 
                              pic):
    
    start_time = time.time()
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭

    # 모델을 통한 마스크 감지
    model_output = model(frame, iou=box_threshold, classes=0, verbose=False)
    if not model_output or not model_output[0].masks:
        cv2.imshow('result', np.full_like(frame, [0, 255, 0], dtype=np.uint8))
        return 0  # 마스크 감지 실패 시 함수 종료

    mask_results = model_output[0].masks.data.cpu()
    
    target_mask_found = False
    processed_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # 초기화

    for i in range(mask_results.shape[0]):
        mask = mask_results[i]
        mask = np.array(mask)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # 마스크 확장
        kernel = np.ones((1, 1), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 확장된 마스크의 내부 영역 채우기
        _, binary_mask = cv2.threshold(dilated_mask, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 각 영역(컨투어)에 대해 x, y 좌표가 포함되는지 확인
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                cv2.drawContours(processed_mask, [contour], -1, 255, thickness=cv2.FILLED)
                target_mask_found = True
                break
        
        if target_mask_found:
            break

    # 마스크가 감지되지 않은 경우 초록색 배경으로 설정
    if target_mask_found:
        # 가우시안 블러로 경계 부드럽게 처리
        #blurred_mask = cv2.GaussianBlur(processed_mask, (1, 1), 0)
        #alpha_mask = blurred_mask / 255.0  # 정규화

        # 원본 frame과 초록색 배경을 픽셀별로 블렌딩
        frame_masked = cv2.bitwise_and(frame, frame, mask=processed_mask)
        background = np.full_like(frame, [0, 255, 0], dtype=np.uint8)
        background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(processed_mask))
        processed_mask = processed_mask / 255.0
        final_frame = (frame_masked * processed_mask[..., None] + background_masked * (1 - processed_mask[..., None])).astype(np.uint8)
    else:
        final_frame = np.full_like(frame, [0, 255, 0], dtype=np.uint8)
        
    ##########################################################################################
    
    final_frame = chroma_key_func(final_frame, pic)
    cv2.imshow('result', final_frame)

    ##########################################################################################

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time

##########################################################################################

def camera_human_seg_old(model, 
                              cap, 
                              x,
                              y, 
                              box_threshold, 
                              pic):
    
    start_time = time.time()
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭

    # 모델을 통한 마스크 감지
    model_output = model(frame, iou=box_threshold, classes=0, verbose=False)
    if not model_output or not model_output[0].masks:
        cv2.imshow('result', np.full_like(frame, [0, 255, 0], dtype=np.uint8))
        return 0  # 마스크 감지 실패 시 함수 종료

    mask_results = model_output[0].masks.data.cpu()
    
    target_mask_found = False
    processed_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # 초기화

    for i in range(mask_results.shape[0]):
        mask = mask_results[i]
        mask = np.array(mask)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # 마스크 확장
        kernel = np.ones((1, 1), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 확장된 마스크의 내부 영역 채우기
        _, binary_mask = cv2.threshold(dilated_mask, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 각 영역(컨투어)에 대해 x, y 좌표가 포함되는지 확인
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                cv2.drawContours(processed_mask, [contour], -1, 255, thickness=cv2.FILLED)
                target_mask_found = True
                break
        
        if target_mask_found:
            break

    # 마스크가 감지되지 않은 경우 초록색 배경으로 설정
    if target_mask_found:
        # 가우시안 블러로 경계 부드럽게 처리
        #blurred_mask = cv2.GaussianBlur(processed_mask, (1, 1), 0)
        #alpha_mask = blurred_mask / 255.0  # 정규화

        # 원본 frame과 초록색 배경을 픽셀별로 블렌딩
        frame_masked = cv2.bitwise_and(frame, frame, mask=processed_mask)
        background = np.full_like(frame, [0, 255, 0], dtype=np.uint8)
        background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(processed_mask))
        processed_mask = processed_mask / 255.0
        final_frame = (frame_masked * processed_mask[..., None] + background_masked * (1 - processed_mask[..., None])).astype(np.uint8)
    else:
        final_frame = np.full_like(frame, [0, 255, 0], dtype=np.uint8)
        
    ##########################################################################################
    
    target_pic = final_frame
    chroma_key_range = ([0, 1, 0, 0], [0, 255, 0, 255])
    # 크로마키 영역 제거 및 투명 부분 제거
    image_target_pic_no_green = remove_green_screen(target_pic, chroma_key_range)
    image_target_pic_cropped = crop_transparent_area(image_target_pic_no_green)
    # 비투명한 부분에만 블러 처리
    #image_target_pic_blurred = blur_non_transparent_area(image_target_pic_cropped, blur_size=(5, 5))
    image_target_pic_blurred = image_target_pic_cropped

    # 이미지 어둡게 처리
    image_target_pic_darkened = darken_image(image_target_pic_blurred, factor=0.8)
    # 이미지의 가장자리를 부드럽게 만들기
    image_target_pic_postprocessed = soften_edges(image_target_pic_darkened, edge_width=51)
    # # 경계 투명화 적용
    # image_hee_faded = fade_object_edges(image_hee_cropped, fade_width=100)
    # 이미지 합성 및 저장
    result_image = combine_images(pic, image_target_pic_postprocessed, blur_size=1)

    # 세피아 효과 적용
    sepia_image = apply_sepia_effect(result_image)
    # 손상 효과 적용
    damaged_image = apply_film_damage_effect(sepia_image)
    # 가장자리 손상 효과 적용
    final_frame = apply_edge_damage_effect(damaged_image)

    cv2.imshow('result', final_frame)
    cv2.imwrite('result.jpg', final_frame)
    final_frame = None

    ##########################################################################################

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time

##########################################################################################






