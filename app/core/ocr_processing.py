import cv2
import numpy as np
import pytesseract
from model.table import Table
import math
import re

def gaussian_score(x, mean, sigma):
    """
    x가 mean에 가까울수록 1에, 멀어질수록 0에 가까운 점수를 반환.
    """
    return math.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def compute_normality_score(text):
    """
    OCR로 추출된 텍스트의 정상성을 두 가지 기준으로 평가:
      1) 평균 단어 길이: 토큰(알파벳 연속 문자열)의 평균 길이가 4.7에 가까울수록 정상적.
         (즉, 약 4.7개의 글자마다 띄어쓰기가 있어야 함)
      2) 모음:자음 비율: 이상적인 비율은 1:2 (0.5)로, 이 값에 가까울수록 정상적.
    
    각 기준은 가우시안 함수를 통해 0~1 사이 점수를 얻으며, 두 점수를 동일 가중치(0.5)로 결합하여
    최종 점수가 1을 넘지 않도록 함.
    """
    # 알파벳으로 이루어진 토큰(단어)들을 추출
    tokens = re.findall(r'[A-Za-z]+', text)
    if not tokens:
        return 0.0  # 평가할 단어가 없으면 비정상으로 처리

    # [1] 평균 단어 길이 평가
    avg_length = sum(len(token) for token in tokens) / len(tokens)
    # 이상적인 평균 길이는 5, sigma는 2.0 (경우에 따라 조정 가능)
    score_length = gaussian_score(avg_length, 5, sigma=2.0)

    # [2] 모음:자음 비율 평가
    # 짧은(길이 1) 토큰은 평가에서 제외하여, 충분한 길이의 단어만 사용
    ratio_list = []
    for token in tokens:
        if len(token) < 2:
            continue
        vowels = sum(1 for ch in token if ch.lower() in "aeiou")
        consonants = sum(1 for ch in token if ch.isalpha() and ch.lower() not in "aeiou")
        if consonants == 0:
            continue  # 자음이 전혀 없는 경우는 제외
        ratio_list.append(vowels / consonants)
    
    if ratio_list:
        avg_ratio = sum(ratio_list) / len(ratio_list)
        # 이상적인 비율은 0.5, sigma는 0.5 (경우에 따라 조정 가능)
        score_ratio = gaussian_score(avg_ratio, 0.5, sigma=0.5)
    else:
        score_ratio = 0.0  # 평가 가능한 단어가 없으면 0으로 처리

    # 각 점수에 0.5의 가중치를 주어 최종 정상성 점수를 산출 (최대 1)
    final_score = 0.5 * score_length + 0.5 * score_ratio
    return final_score

def preprocess_cell_image(cell_img, crop_px=4, rescale_factor=1.2):
    h, w = cell_img.shape[:2]
    crop_top = crop_px if h > 2 * crop_px else 0
    crop_bottom = h - crop_px if h > 2 * crop_px else h
    crop_left = crop_px if w > 2 * crop_px else 0
    crop_right = w - crop_px if w > 2 * crop_px else w
    cropped = cell_img[crop_top:crop_bottom, crop_left:crop_right]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, binary = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY_INV)

    black_pixels = np.sum(binary == 0)
    white_pixels = np.sum(binary == 255)
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)

    new_w = int(binary.shape[1] * rescale_factor)
    new_h = int(binary.shape[0] * rescale_factor)
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed

def should_rotate(image, canny_thresh1=50, canny_thresh2=150, hough_threshold=30,
                  min_line_length_ratio=0.7, max_line_gap=5):
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

def ocr_cell(cell: Table.Cell, original_img, normal_cell_height=None):
    """
    Table.Cell 객체의 속성(x_start, y_start, x_end, y_end)을 이용해 셀 영역을 크롭하고,
    전처리 및 OCR 수행 후 텍스트를 반환합니다.
    """
    # 셀 영역 크롭
    x, y = cell.x_start, cell.y_start
    w = cell.x_end - cell.x_start
    h = cell.y_end - cell.y_start
    cell_img = original_img[y:y+h, x:x+w]
    
    preprocessed = preprocess_cell_image(cell_img, crop_px=4, rescale_factor=2)
    
    unique, counts = np.unique(preprocessed, return_counts=True)
    if counts.max() / preprocessed.size > 0.98:
        return ""
    
    if cell.area is not None:
        # cell.area는 ((min_r, max_r), (min_c, max_c)) 형태로 저장되어 있음
        min_r, max_r = cell.area[0]
        y_span = max_r - min_r + 1
    else:
        y_span = 1
    
    config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'
    if y_span < 3:
        text = pytesseract.image_to_string(preprocessed, config=config)
    else:
        text_horiz = pytesseract.image_to_string(preprocessed, config=config)
        if should_rotate(preprocessed):
            corrected = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)
            text_rotated = pytesseract.image_to_string(corrected, config=config)
        else:
            text_rotated = ""
        score1 = compute_normality_score(text_horiz.strip())
        score2 = compute_normality_score(text_rotated.strip())
        text = text_rotated if score2 > score1 else text_horiz
    
    return text.strip()

def extract_table_data(table: Table):
    """
    Table 객체의 모든 셀에 대해 OCR 수행 후, 해당 셀의 data 속성에 텍스트를 저장합니다.
    Table의 y_coords를 이용해 정상 셀 높이를 추정합니다.
    """
    if len(table.y_coords) > 1:
        normal_cell_height = table.y_coords[1] - table.y_coords[0]
    else:
        normal_cell_height = None
    
    for cell in table.cells:
        text = ocr_cell(cell, table.table_image, normal_cell_height)
        cell.data = text
