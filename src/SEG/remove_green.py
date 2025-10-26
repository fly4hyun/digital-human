import cv2
import numpy as np

def remove_chroma_key(image, color_range):
    """ 크로마키 색상을 제거하고 해당 영역을 투명하게 처리 """
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

# 이미지 로드
image = cv2.imread('hee.png')  # 이미지 경로를 지정하세요

# 초록색 크로마키 색상 범위 정의 ([0, 1, 0] ~ [0, 255, 0] 범위)
chroma_key_range = ([0, 1, 0, 0], [0, 255, 0, 255])

# 크로마키 처리
transparent_image = remove_chroma_key(image, chroma_key_range)
cv2.imwrite('cropped_image_0.png', transparent_image)  # PNG 포맷으로 저장
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

# 이미지 로드 및 크로마키 처리
# [크로마키 처리 코드]
# 예: transparent_image = remove_chroma_key(image, chroma_key_range)

# 투명 영역을 제외하고 이미지 자르기
cropped_image = crop_transparent_area(transparent_image)

# 결과 이미지 저장
cv2.imwrite('cropped_image.png', cropped_image)  # PNG 포맷으로 저장
