# app/core/ocr_processing.py
import cv2
import numpy as np
import pytesseract
import time
import functools
from model.table import Table

def crop_border_dynamic(image, extra_crop=2, similarity_threshold=15):
    """
    이미지의 상하좌우 경계에서 배경과 비슷한 영역을 감지하여 잘라냄.
    픽셀 값이 특정 임계값(similarity_threshold) 이내로 유사한 경우 경계로 판단.
    
    Parameters:
        image (np.ndarray): 입력 이미지
        extra_crop (int): 경계 감지 후 추가로 잘라낼 픽셀 수 (기본값 2)
        similarity_threshold (int): 경계 유사도를 판단하는 임계값 (기본값 15)

    Returns:
        np.ndarray: 테두리가 제거된 이미지
    """
    h, w = image.shape[:2]

    def border_thickness(ref_color, axis, reverse=False):
        """
        경계 두께를 측정하는 내부 함수.
        """
        indices = range(h) if axis == 0 else range(w)
        indices = reversed(indices) if reverse else indices
        return sum(np.linalg.norm(np.mean(image[i, :] if axis == 0 else image[:, i], axis=0) - ref_color) < similarity_threshold for i in indices)

    top_crop = min(border_thickness(np.mean(image[0, :], axis=0), 0) + extra_crop, h)
    bottom_crop = max(h - (border_thickness(np.mean(image[-1, :], axis=0), 0, True) + extra_crop), 0)
    left_crop = min(border_thickness(np.mean(image[:, 0], axis=0), 1) + extra_crop, w)
    right_crop = max(w - (border_thickness(np.mean(image[:, -1], axis=0), 1, True) + extra_crop), 0)

    return image[top_crop:bottom_crop, left_crop:right_crop]

def preprocess_cell_image(cell_img):
    """
    OCR 성능 향상을 위해 셀 이미지를 전처리하는 함수.
    1. 테두리 자르기 (crop_border_dynamic)
    2. 그레이스케일 변환
    3. 크기 확대 (OCR 인식률 개선)
    4. 정규화 후 이진화 (흑백 변환)
    5. 배경이 검은색이 많으면 반전
    6. 모폴로지 변환으로 노이즈 제거

    Parameters:
        cell_img (np.ndarray): 입력 셀 이미지

    Returns:
        np.ndarray: 전처리된 바이너리 이미지
    """
    cropped = crop_border_dynamic(cell_img)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX), 127, 255, cv2.THRESH_BINARY_INV)
    
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

def crop_masked_area(x_coords, y_coords, cell: Table.Cell, cell_img):
    """
    병합된 셀에서 원래 각 개별 셀의 영역을 마스킹하여 OCR 성능 향상.

    Parameters:
        x_coords (list): x 좌표 리스트
        y_coords (list): y 좌표 리스트
        cell (Table.Cell): OCR 대상 셀 객체
        cell_img (np.ndarray): 크롭된 셀 이미지

    Returns:
        np.ndarray: 마스킹된 이미지
    """
    h, w = cell_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    ox, oy, extra = cell.x_start, cell.y_start, 5

    for row, col in cell.unmerged_coords:
        left, right = max(x_coords[col] - ox - extra, 0), min(x_coords[col+1] - ox + extra, w)
        top, bottom = max(y_coords[row] - oy - extra, 0), min(y_coords[row+1] - oy + extra, h)
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

    return cv2.inpaint(cell_img, mask, 1, cv2.INPAINT_TELEA)

def should_rotate(image, hough_threshold=30, min_line_length_ratio=0.7, max_line_gap=5):
    """
    이미지 내에서 주요 선들을 분석하여 회전이 필요한지 판단.
    
    1. 허프 변환(Hough Transform)을 사용하여 직선을 감지.
    2. 수직선(45도 이상)과 수평선(45도 이하)의 개수를 비교.
    3. 수직선이 더 많으면 회전이 필요하다고 판단하여 True 반환.

    Parameters:
        image (np.ndarray): 입력 이미지 (이진화된 테이블 이미지)
        hough_threshold (int): 허프 변환의 최소 투표 개수 (기본값 30)
        min_line_length_ratio (float): 감지할 최소 선 길이 (기본값 0.7, 이미지 높이의 70%)
        max_line_gap (int): 선의 최대 끊긴 거리 (기본값 5)

    Returns:
        bool: 이미지가 회전이 필요하면 True, 그렇지 않으면 False
    """
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=hough_threshold,
                            minLineLength=int(image.shape[0] * min_line_length_ratio), maxLineGap=max_line_gap)

    if not lines:
        return False  # 직선이 감지되지 않으면 회전 불필요

    count_vertical, count_horizontal = 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))  # 선의 각도 계산

        if angle > 45:
            count_vertical += 1  # 수직선 개수 증가
        else:
            count_horizontal += 1  # 수평선 개수 증가

    return count_vertical > count_horizontal  # 수직선이 많으면 회전 필요

def merge_boxes(boxes, threshold=10):
    """
    가까운 바운딩 박스를 병합.

    Parameters:
        boxes (list): [(x, y, w, h), ...] 형식의 박스 리스트
        threshold (int): 병합할 최대 거리 (기본값 10 픽셀)

    Returns:
        list: 병합된 박스 리스트
    """
    changed = True
    while changed:
        changed, new_boxes, used = False, [], [False] * len(boxes)

        for i, box_a in enumerate(boxes):
            if used[i]:  # 이미 병합됨
                continue

            for j, box_b in enumerate(boxes[i+1:], start=i+1):
                if used[j]:  
                    continue

                # 박스 간 거리 계산
                gap_x = max(0, box_b[0] - (box_a[0] + box_a[2])) if box_a[0] < box_b[0] else max(0, box_a[0] - (box_b[0] + box_b[2]))
                gap_y = max(0, box_b[1] - (box_a[1] + box_a[3])) if box_a[1] < box_b[1] else max(0, box_a[1] - (box_b[1] + box_b[3]))

                if gap_x < threshold and gap_y < threshold:
                    # 박스 병합
                    box_a = (
                        min(box_a[0], box_b[0]),
                        min(box_a[1], box_b[1]),
                        max(box_a[0] + box_a[2], box_b[0] + box_b[2]) - min(box_a[0], box_b[0]),
                        max(box_a[1] + box_a[3], box_b[1] + box_b[3]) - min(box_a[1], box_b[1])
                    )
                    used[j], changed = True, True  # 병합된 박스 표시

            new_boxes.append(box_a)  

        boxes = new_boxes  

    return boxes

def find_bounding_boxes(binary_img, merge_threshold=10):
    """
    바이너리 이미지에서 개별 글자 또는 객체의 바운딩 박스를 찾고 병합.
    
    1. 윤곽선(contour) 검출을 통해 객체 영역을 찾음.
    2. 예외 처리: 테이블 전체 크기와 거의 같은 큰 박스는 제외.
    3. 서로 가까운 바운딩 박스를 병합하여 하나의 박스로 합침.
    4. 최종 바운딩 박스를 Y 좌표 기준 정렬 (같은 줄에 있는 글자들이 그룹화되도록).

    Parameters:
        binary_img (np.ndarray): 입력 이진화 이미지 (OCR 전처리를 위한 바이너리 이미지)
        merge_threshold (int): 병합할 최대 거리 (기본값 10 픽셀)

    Returns:
        list: [(x, y, w, h), ...] 형식의 바운딩 박스 리스트
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary_img.shape[:2]

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 예외 처리: 이미지 전체를 차지하는 박스는 무시
        if x <= 2 and y <= 2 and (x + w) >= (w - 2) and (y + h) >= (h - 2):
            continue

        boxes.append((x, y, w, h))  # 바운딩 박스 추가

    # 가까운 바운딩 박스를 병합
    merged_boxes = merge_boxes(boxes, merge_threshold)

    # 박스를 Y 좌표 기준 정렬 (동일한 줄에 있는 글자들을 인식하기 위함)
    return sorted(merged_boxes, key=functools.cmp_to_key(lambda a, b: a[0] - b[0] if abs(a[1] - b[1]) <= 10 else a[1] - b[1]))


def ocr_cell(table: Table, cell: Table.Cell, original_img):
    """
    셀 이미지를 OCR로 처리하여 텍스트를 추출.
    1. 셀을 크롭하고 병합된 경우 마스킹 적용
    2. OCR 성능을 높이기 위해 이미지 전처리
    3. 회전이 필요한 경우 자동으로 감지하여 보정
    4. 최종 OCR을 실행하여 텍스트를 반환

    Parameters:
        table (Table): 테이블 객체
        cell (Table.Cell): OCR을 수행할 개별 셀
        original_img (np.ndarray): 원본 테이블 이미지

    Returns:
        str: OCR로 추출된 텍스트
    """
    x, y, w, h = cell.x_start, cell.y_start, cell.x_end - cell.x_start, cell.y_end - cell.y_start
    cell_img = original_img[y:y+h, x:x+w]
    
    if cell.is_merged and cell.unmerged_coords:
        cell_img = crop_masked_area(table.x_coords, table.y_coords, cell, cell_img)

    preprocessed = preprocess_cell_image(cell_img)

    if np.max(np.unique(preprocessed, return_counts=True)[1]) / preprocessed.size > 0.995:
        return ""

    max_r, min_r = cell.area[0][1], cell.area[0][0]
    y_span = max_r - min_r + 1
    is_rotated = cell.is_merged and y_span > 2 and should_rotate(preprocessed)

    if is_rotated:
        preprocessed = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)

    final_boxes = find_bounding_boxes(preprocessed, merge_threshold=20)
    
    if is_rotated and any(bh > bw for _, _, bw, bh in final_boxes):
        preprocessed = cv2.rotate(preprocessed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        final_boxes = find_bounding_boxes(preprocessed, merge_threshold=20)

    config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'
    extracted_texts = []

    for bx, by, bw, bh in final_boxes:
        x1 = max(bx - 3, 0)
        y1 = max(by - 3, 0)
        x2 = min(bx + bw + 3, preprocessed.shape[1])
        y2 = min(by + bh + 3, preprocessed.shape[0])

        if x2 > x1 and y2 > y1:
            cropped_box = preprocessed[y1:y2, x1:x2]
            text = pytesseract.image_to_string(cropped_box, config=config).strip()
            if text:
                extracted_texts.append(text)

    return " ".join(extracted_texts)

def print_progress(progress, bar_length=50):
    """
    진행 상태를 프로그레스 바 형태로 출력.
    """
    filled_length = int(bar_length * progress / 100)
    print(f"\rProgress: |{'#' * filled_length}{'-' * (bar_length - filled_length)}| {progress}%", end="", flush=True)


def extract_table_data(table: Table):
    """
    테이블의 모든 셀을 OCR로 처리하여 데이터를 추출.
    진행률을 표시하면서 작업을 수행.
    """
    total = len(table.cells)
    current_progress = 0
    
    for idx, cell in enumerate(table.cells):
        try:
            cell.data = ocr_cell(table, cell, table.table_image)
        except:
            cell.data = ""

        target_progress = int((idx + 1) / total * 100)
        for p in range(current_progress + 1, target_progress + 1):
            print_progress(p)
            time.sleep(0.02)
        current_progress = target_progress
    print()
