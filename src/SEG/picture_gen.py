import cv2
import numpy as np

##########################################################################################

def remove_green_screen(image, color_range):
    # 이미지를 RGBA로 변환
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 크로마키 색상 범위 정의
    lower, upper = color_range
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # 크로마키 색상 영역을 찾고 투명하게 처리
    mask = cv2.inRange(rgba_image, lower, upper)
    rgba_image[mask > 0] = [0, 0, 0, 0]

    return rgba_image

def crop_transparent_area(image):
    """ 투명한 영역을 제외하고 이미지를 자름 """
    # 알파 채널에서 투명하지 않은 부분의 경계 찾기
    alpha_channel = image[:, :, 3]
    _, thresh = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # 경계를 찾고 이미지 자르기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        return image  # 경계를 찾지 못한 경우 원본 이미지 반환

def blur_non_transparent_area(image, blur_size=(5, 5)):
    """ 이미지의 비투명한 부분에만 블러 처리 적용 """
    if image.shape[2] != 4:
        raise ValueError("Image must have 4 channels (RGBA).")

    # RGBA 이미지에서 RGB 채널과 알파 채널 분리
    rgb_channels = image[:, :, :3]
    alpha_channel = image[:, :, 3]

    # 비투명한 부분만 블러 처리
    blurred_rgb = cv2.GaussianBlur(rgb_channels, blur_size, 0)
    blurred_rgb[alpha_channel == 0] = 0  # 투명한 부분은 블러 처리하지 않음

    # 블러 처리된 RGB 채널과 원본 알파 채널 결합
    blurred_image = np.dstack((blurred_rgb, alpha_channel))
    return blurred_image

##########################################################################################

def darken_image(image, factor=0.7):
    """ 이미지의 RGB 채널만 어둡게 만드는 함수 """
    if image.shape[2] == 4:
        # RGBA 이미지의 경우, RGB와 알파 채널 분리
        rgb_channels = image[:, :, :3]
        alpha_channel = image[:, :, 3]

        # RGB 채널만 어둡게 처리
        darkened_rgb = cv2.multiply(rgb_channels, np.array([factor]))

        # 어둡게 처리된 RGB 채널과 원본 알파 채널을 다시 결합
        darkened_image = np.dstack((darkened_rgb, alpha_channel))
        return darkened_image
    else:
        # RGB 이미지의 경우, 바로 어둡게 처리
        return cv2.multiply(image, np.array([factor]))

def soften_edges(image, edge_width=5):
    """ 이미지의 가장자리를 부드럽게 만드는 함수 """
    if image.shape[2] != 4:
        raise ValueError("Image must have 4 channels (RGBA).")

    # 알파 채널 추출
    alpha_channel = image[:, :, 3]

    # 경계 영역을 부드럽게 처리
    kernel_size = (edge_width, edge_width)
    blurred_alpha = cv2.GaussianBlur(alpha_channel, kernel_size, 0)
    alpha_channel = cv2.addWeighted(alpha_channel, 0.5, blurred_alpha, 0.5, 0)

    # 수정된 알파 채널 적용
    image[:, :, 3] = alpha_channel
    return image

# def fade_object_edges(image, fade_width=10, edge_dilation=5):
#     """ 이미지의 객체 경계를 넓게 투명하게 만드는 함수 """
#     if image.shape[2] != 4:
#         raise ValueError("Image must have 4 channels (RGBA).")

#     alpha_channel = image[:, :, 3]
#     mask = np.zeros_like(alpha_channel)

#     # 알파 채널에서 객체 영역 추출
#     mask[alpha_channel > 0] = 255

#     # 객체 외곽을 추출하고 외곽선 확장
#     edges = cv2.Canny(mask, 50, 150)
#     dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=edge_dilation)
#     fade_mask = cv2.GaussianBlur(dilated_edges, (fade_width*2+1, fade_width*2+1), 0)

#     # 페이드 마스크 적용
#     image[:, :, 3] = alpha_channel * (1 - fade_mask / 255) + fade_mask

#     return image

def combine_images(background_img, foreground_img, blur_size=5):
    """ 배경 이미지에 전경 이미지를 부드럽게 합성 """
    fg_height, fg_width = foreground_img.shape[:2]
    bg_height, bg_width = background_img.shape[:2]

    # 전경 이미지가 4 채널(투명도 포함)을 갖고 있는지 확인
    if foreground_img.shape[2] == 4:
        alpha_channel = foreground_img[:, :, 3]
    else:
        # 알파 채널이 없는 경우, 완전 불투명한 알파 채널 추가
        alpha_channel = np.ones((fg_height, fg_width), dtype=foreground_img.dtype) * 255

    # 가우시안 블러 적용
    alpha_channel = cv2.GaussianBlur(alpha_channel, (blur_size, blur_size), 0)

    # 전경 이미지가 배경 이미지보다 큰 경우 크기 조절
    if fg_height > bg_height or fg_width > bg_width:
        scale_factor = min(bg_width / fg_width, bg_height / fg_height)
        new_size = (int(fg_width * scale_factor), int(fg_height * scale_factor))
        resized_foreground = cv2.resize(foreground_img, new_size, interpolation=cv2.INTER_AREA)
        resized_alpha_channel = cv2.resize(alpha_channel, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized_foreground = foreground_img
        resized_alpha_channel = alpha_channel

    # 합성할 위치 계산 (하단 오른쪽)
    x_offset = bg_width - resized_foreground.shape[1]
    y_offset = bg_height - resized_foreground.shape[0]

    # 배경 이미지에 전경 이미지 합성
    for c in range(0, 3):
        background_img[y_offset:y_offset+resized_foreground.shape[0], x_offset:x_offset+resized_foreground.shape[1], c] = \
            resized_foreground[:, :, c] * (resized_alpha_channel / 255.0) + \
            background_img[y_offset:y_offset+resized_foreground.shape[0], x_offset:x_offset+resized_foreground.shape[1], c] * (1 - resized_alpha_channel / 255.0)

    return background_img


##########################################################################################

def apply_sepia_effect(image):
    """ 세피아 효과 적용 """
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    sepia_filter = np.array([[0.272, 0.534, 0.131, 0],
                             [0.349, 0.686, 0.168, 0],
                             [0.393, 0.769, 0.189, 0],
                             [0, 0, 0, 1]])
    sepia_image = cv2.transform(converted, sepia_filter)
    return cv2.cvtColor(sepia_image, cv2.COLOR_BGRA2BGR)

def apply_film_damage_effect(image, noise_intensity=50, scratch_intensity=0.3, dust_particles=1000):
    """ 필름 손상 효과 적용 """
    rows, cols, _ = image.shape

    # 랜덤 잡음 추가
    noise = np.random.randint(0, noise_intensity, (rows, cols, 1), dtype=np.uint8)
    noise_img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    image = cv2.add(image, noise_img)

    # 랜덤 스크래치 추가
    for _ in range(np.random.randint(1, 10)):
        x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
        x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
        image = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 먼지 입자 추가
    for _ in range(dust_particles):
        x, y = np.random.randint(0, cols), np.random.randint(0, rows)
        image = cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

    return image

def apply_edge_damage_effect(image, noise_intensity=100, extra_scratches=10, edge_width=50):
    """ 이미지 외각에 더 심한 손상 효과 적용 """
    rows, cols, _ = image.shape

    # 이미지의 가장자리 생성
    edge_mask = np.zeros((rows, cols), dtype=np.uint8)
    edge_mask[edge_width:-edge_width, edge_width:-edge_width] = 255
    edge_mask = cv2.bitwise_not(edge_mask)

    # 가장자리에 랜덤 잡음 추가
    noise = np.random.randint(0, noise_intensity, (rows, cols, 1), dtype=np.uint8)
    noise_img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    noise_img[edge_mask == 0] = 0  # 가장자리에만 잡음 적용
    image = cv2.add(image, noise_img)

    # 가장자리에 추가 스크래치 추가
    for _ in range(extra_scratches):
        x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
        x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
        image = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return image

##########################################################################################

# 이미지 로드 및 처리
image_hee = cv2.imread('hee.png', cv2.IMREAD_UNCHANGED)
chroma_key_range = ([0, 1, 0, 0], [0, 255, 0, 255])
image_kimgoo = cv2.imread('kimgoo.jpg')

##########################################################################################

# 크로마키 영역 제거 및 투명 부분 제거
image_hee_no_green = remove_green_screen(image_hee, chroma_key_range)
image_hee_cropped = crop_transparent_area(image_hee_no_green)
# 비투명한 부분에만 블러 처리
image_hee_blurred = blur_non_transparent_area(image_hee_cropped, blur_size=(5, 5))

##########################################################################################

# 이미지 어둡게 처리
image_hee_darkened = darken_image(image_hee_blurred, factor=0.8)
# 이미지의 가장자리를 부드럽게 만들기
image_hee_postprocessed = soften_edges(image_hee_darkened, edge_width=51)
# # 경계 투명화 적용
# image_hee_faded = fade_object_edges(image_hee_cropped, fade_width=100)
# 이미지 합성 및 저장
result_image = combine_images(image_kimgoo, image_hee_postprocessed, blur_size=1)
cv2.imwrite('result.jpg', result_image)

##########################################################################################

# 세피아 효과 적용
sepia_image = apply_sepia_effect(result_image)
# 손상 효과 적용
damaged_image = apply_film_damage_effect(sepia_image)
# 가장자리 손상 효과 적용
damaged_image_with_edges = apply_edge_damage_effect(damaged_image)

cv2.imwrite('result_old.jpg', damaged_image_with_edges)

##########################################################################################
