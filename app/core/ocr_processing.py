import cv2
import numpy as np
import pytesseract
import time
import functools
from model.table import Table

def crop_border_dynamic(image, extra_crop=2, similarity_threshold=15):
    """
    이미지의 상하좌우 경계에서 각 방향의 평균 색상이 기준 색상과 비슷한 정도를 측정하여
    경계선 두께를 구한 후, (두께 + extra_crop) 만큼 잘라낸 이미지를 반환합니다.
    
    Parameters:
        image (np.ndarray): 입력 이미지 (컬러 이미지)
        extra_crop (int): 측정된 경계선 두께에 추가로 잘라낼 픽셀 수 (기본값 2)
        similarity_threshold (float): 평균 색상의 유사도를 판단하는 임계치 (기본값 15)
                                      (픽셀 값 차이의 유클리드 거리가 이 값 이하면 동일한 색상으로 판단)
    
    Returns:
        cropped (np.ndarray): 경계선 영역을 잘라낸 결과 이미지
    """
    h, w = image.shape[:2]

    # Top border
    ref_top = np.mean(image[0, :], axis=0)
    top_thickness = 0
    for i in range(h):
        row_mean = np.mean(image[i, :], axis=0)
        if np.linalg.norm(row_mean - ref_top) < similarity_threshold:
            top_thickness += 1
        else:
            break

    # Bottom border
    ref_bottom = np.mean(image[h - 1, :], axis=0)
    bottom_thickness = 0
    for i in range(h - 1, -1, -1):
        row_mean = np.mean(image[i, :], axis=0)
        if np.linalg.norm(row_mean - ref_bottom) < similarity_threshold:
            bottom_thickness += 1
        else:
            break

    # Left border
    ref_left = np.mean(image[:, 0], axis=0)
    left_thickness = 0
    for j in range(w):
        col_mean = np.mean(image[:, j], axis=0)
        if np.linalg.norm(col_mean - ref_left) < similarity_threshold:
            left_thickness += 1
        else:
            break

    # Right border
    ref_right = np.mean(image[:, w - 1], axis=0)
    right_thickness = 0
    for j in range(w - 1, -1, -1):
        col_mean = np.mean(image[:, j], axis=0)
        if np.linalg.norm(col_mean - ref_right) < similarity_threshold:
            right_thickness += 1
        else:
            break

    # 각 방향에서 (경계선 두께 + extra_crop) 만큼 잘라낼 영역 계산
    top_crop = top_thickness + extra_crop
    bottom_crop = h - (bottom_thickness + extra_crop)
    left_crop = left_thickness + extra_crop
    right_crop = w - (right_thickness + extra_crop)

    # 범위 조정 (이미지 범위를 벗어나지 않도록)
    top_crop = min(top_crop, h)
    bottom_crop = max(bottom_crop, 0)
    left_crop = min(left_crop, w)
    right_crop = max(right_crop, 0)

    cropped = image[top_crop:bottom_crop, left_crop:right_crop]
    return cropped

def preprocess_cell_image(cell_img):
    cropped = crop_border_dynamic(cell_img)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    resized_gray = cv2.resize(gray, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_LINEAR)
    
    normalized = cv2.normalize(resized_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, binary = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY_INV)

    black_pixels = np.sum(binary == 0)
    white_pixels = np.sum(binary == 255)
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # h, w = rotated.shape
    # final_image = cv2.resize(rotated, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_LINEAR)
    return final_image

def get_box_gap(boxA, boxB):
    # boxA, boxB: (x, y, w, h)
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    if x1 + w1 < x2:
        gap_x = x2 - (x1 + w1)
    elif x2 + w2 < x1:
        gap_x = x1 - (x2 + w2)
    else:
        gap_x = 0

    if y1 + h1 < y2:
        gap_y = y2 - (y1 + h1)
    elif y2 + h2 < y1:
        gap_y = y1 - (y2 + h2)
    else:
        gap_y = 0

    return gap_x, gap_y

def union_box(boxA, boxB):
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    new_x = min(x1, x2)
    new_y = min(y1, y2)
    new_x2 = max(x1 + w1, x2 + w2)
    new_y2 = max(y1 + h1, y2 + h2)
    return (new_x, new_y, new_x2 - new_x, new_y2 - new_y)

def merge_boxes(boxes, threshold=10):
    # 더 이상 병합할 수 없을 때까지 반복
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            box_a = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                box_b = boxes[j]
                gap_x, gap_y = get_box_gap(box_a, box_b)
                if gap_x < threshold and gap_y < threshold:
                    box_a = union_box(box_a, box_b)
                    used[j] = True
                    changed = True
            new_boxes.append(box_a)
        boxes = new_boxes
    return boxes

def draw_and_merge_bounding_boxes(binary_img, merge_threshold=10):
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = binary_img.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 예외 박스 제거: 좌상단이 0에 가깝고 우하단이 이미지 크기에 가까운 박스
        if x <= 2 and y <= 2 and (x + w) >= (w_img - 2) and (y + h) >= (h_img - 2):
            continue
        boxes.append((x, y, w, h))
    
    # 예외 박스 제거 후에 가까운 박스들 병합
    merged_boxes = merge_boxes(boxes, threshold=merge_threshold)
    
    # merged_boxes를 좌측 상단 기준으로 정렬
    # 비슷한 높이(즉, y 좌표 차이가 10px 이하)라면 왼쪽(x 값)이 작은 것이 앞으로 오고,
    # 그렇지 않다면 y 값이 작은(높은) 것이 먼저 오도록 정렬합니다.
    def box_compare(a, b):
        y_threshold = 10
        # a, b: (x, y, w, h)
        if abs(a[1] - b[1]) <= y_threshold:
            return a[0] - b[0]
        else:
            return a[1] - b[1]
    
    sorted_boxes = sorted(merged_boxes, key=functools.cmp_to_key(box_compare))
    return sorted_boxes

def should_rotate(image, hough_threshold=30, min_line_length_ratio=0.7, max_line_gap=5):
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

def ocr_cell(cell: Table.Cell, original_img):
    """
    Table.Cell 객체의 속성(x_start, y_start, x_end, y_end)을 이용해 셀 영역을 크롭하고,
    전처리 및 OCR 수행 후 텍스트를 반환합니다.
    """
    # 셀 영역 크롭
    x, y = cell.x_start, cell.y_start
    w = cell.x_end - cell.x_start
    h = cell.y_end - cell.y_start
    cell_img = original_img[y:y+h, x:x+w]
    
    preprocessed = preprocess_cell_image(cell_img)
    
    _, counts = np.unique(preprocessed, return_counts=True)
    if counts.max() / preprocessed.size > 0.98:
        return ""
    
    final_boxes = draw_and_merge_bounding_boxes(preprocessed, merge_threshold=15)
    
    config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'
    h_img, w_img = preprocessed.shape[:2]
    all_texts = []
    
    for box in final_boxes:
        bx, by, bw, bh = box
        margin = 3
        # 박스 크기에 좌우상하 3px씩 패딩 추가 (이미지 범위 내로 제한)
        x1 = max(bx - margin, 0)
        y1 = max(by - margin, 0)
        x2 = min(bx + bw + margin, w_img)
        y2 = min(by + bh + margin, h_img)
        
        cropped_box = preprocessed[y1:y2, x1:x2]
        cropped_box = cv2.rotate(cropped_box, cv2.ROTATE_90_CLOCKWISE) if should_rotate(cropped_box) else cropped_box
        text = pytesseract.image_to_string(cropped_box, config=config)
        if text:
            all_texts.append(text.strip())
    
    final_text = " ".join(all_texts)
    return final_text

def print_progress(progress, bar_length=50):
    """
    progress 값(0~100)에 따라 progress bar를 갱신합니다.
    """
    filled_length = int(bar_length * progress / 100)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    # carriage return(\r)으로 같은 줄에 덮어쓰기
    print(f"\rProgress: |{bar}| {progress}%", end="", flush=True)

def extract_table_data(table: Table):
    """
    Table 객체의 모든 셀에 대해 OCR 수행 후, 해당 셀의 data 속성에 텍스트를 저장하고,
    진행률을 애니메이션 효과와 함께 터미널에 표시합니다.
    """
    total = len(table.cells)
    current_progress = 0
    for idx, cell in enumerate(table.cells):
        try:
            text = ocr_cell(cell, table.table_image)
        except Exception as e:
            text = ""
        cell.data = text

        # 목표 진행률 계산 (0~100)
        target_progress = int((idx + 1) / total * 100)
        # 현재 진행률부터 목표 진행률까지 1씩 증가시키며 애니메이션 효과
        for p in range(current_progress + 1, target_progress + 1):
            print_progress(p)
            time.sleep(0.02)  # 애니메이션 속도 조절 가능
        current_progress = target_progress
    print()  # 작업 완료 후 줄바꿈

