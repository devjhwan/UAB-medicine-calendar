# ocr_processing.py
import cv2
import numpy as np
import pytesseract

def preprocess_cell_image(cell_img, crop_px=4, rescale_factor=1.2):
    """
    셀 이미지에 대해 다음 전처리 단계를 수행:
      1. 상하좌우에서 crop_px 픽셀씩 잘라내어 테두리 제거.
      2. 그레이스케일 변환 후, cv2.normalize()를 이용해 정규화하여 명암 대비 극대화.
      3. 전역 이진화 (임계값 127, THRESH_BINARY_INV) 적용.
      4. 히스토그램 분석 후 색상 반전: 0과 255 픽셀 수 비교.
      5. 리스케일링: 이미지 크기를 rescale_factor 배 확대.
      6. Morphological Closing으로 잡음 제거.
    """
    # 1. 테두리 제거
    h, w = cell_img.shape[:2]
    crop_top = crop_px if h > 2*crop_px else 0
    crop_bottom = h - crop_px if h > 2*crop_px else h
    crop_left = crop_px if w > 2*crop_px else 0
    crop_right = w - crop_px if w > 2*crop_px else w
    cropped = cell_img[crop_top:crop_bottom, crop_left:crop_right]
    
    # 2. 그레이스케일 변환 및 정규화
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # 3. 전역 이진화 (고정 임계값 127, THRESH_BINARY_INV)
    _, binary = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 4. 히스토그램 분석 후 색상 반전
    black_pixels = np.sum(binary == 0)
    white_pixels = np.sum(binary == 255)
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)
    
    # 5. 리스케일링
    new_w = int(binary.shape[1] * rescale_factor)
    new_h = int(binary.shape[0] * rescale_factor)
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 6. Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return closed

def should_rotate(image, canny_thresh1=50, canny_thresh2=150, hough_threshold=30,
                  min_line_length_ratio=0.7, max_line_gap=5):
    """
    전처리된 이진화 이미지에서 HoughLinesP를 이용해 선을 검출하고,
    선 각도 분석을 통해 수직선(각도 > 45°)이 수평선보다 많으면 True를 반환한다.
    """
    lines = cv2.HoughLinesP(image, 1, np.pi/180,
                            threshold=hough_threshold,
                            minLineLength=int(image.shape[0] * min_line_length_ratio),
                            maxLineGap=max_line_gap)
    if lines is None:
        return False

    count_vertical = 0
    count_horizontal = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 45:
            count_vertical += 1
        else:
            count_horizontal += 1

    return count_vertical > count_horizontal

def ocr_cell(cell, original_img):
    """
    주어진 셀(cell) 정보를 바탕으로 원본 이미지에서 해당 셀의 전체 병합 영역을 크롭하고,
    전처리(preprocess_cell_image)를 적용하여 OCR 분석을 수행한다.
    
    - y_span이 1 또는 2인 경우: 전처리된 셀 이미지에 대해 단일 OCR 수행.
    - y_span이 3 이상인 경우: 전처리된 셀 이미지에 대해 OCR을 수행하고,
      허프 변환 기반 기울기 감지 후 필요 시 시계 방향 90도 회전한 이미지에 대해 OCR을 수행하여,
      두 결과 중 더 긴 텍스트를 최종 결과로 선택한다.
    만약 전처리된 이미지의 95% 이상 픽셀이 동일하면 텍스트가 없다고 판단하여 빈 문자열을 반환한다.
    """
    x, y = cell['x_start'], cell['y_start']
    w, h = cell['x_length'], cell['y_length']
    cell_img = original_img[y:y+h, x:x+w]
    
    preprocessed = preprocess_cell_image(cell_img, crop_px=4, rescale_factor=2)
    
    unique, counts = np.unique(preprocessed, return_counts=True)
    if counts.max() / preprocessed.size > 0.95:
        return ""
    
    if cell['y_span'] < 3:
        text = pytesseract.image_to_string(preprocessed, 
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-')
    else:
        text_horiz = pytesseract.image_to_string(preprocessed,
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-')
        if should_rotate(preprocessed):
            corrected = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)
            text_rotated = pytesseract.image_to_string(corrected,
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-')
        else:
            text_rotated = ""
        if len(text_rotated.strip()) > len(text_horiz.strip()):
            text = text_rotated
        else:
            text = text_horiz
    return text.strip()

def extract_table_data(image_path, cells):
    """
    주어진 원본 이미지와 활성 셀 리스트(cells)를 사용해 각 셀별 OCR 데이터를 추출하고,
    해당 셀 객체의 'data' 필드에 저장한다.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    for cell in cells:
        text = ocr_cell(cell, original_img)
        cell['data'] = text
