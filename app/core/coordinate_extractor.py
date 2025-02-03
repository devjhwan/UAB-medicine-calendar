# coordinate_extractor.py
import cv2
import numpy as np

def filter_close_coordinates(coords, threshold=10):
    """
    주어진 좌표 리스트에서 서로 threshold 픽셀 이내로 가까운 값들을 그룹화하여,
    각 그룹의 중앙값을 대표값으로 반환하는 함수.
    """
    if not coords:
        return []
    coords = sorted(coords)
    filtered = []
    group = [coords[0]]
    
    for coord in coords[1:]:
        if abs(coord - group[-1]) < threshold:
            group.append(coord)
        else:
            filtered.append(int(np.mean(group)))
            group = [coord]
    filtered.append(int(np.mean(group)))
    return filtered

def get_filtered_coordinates(image_path, threshold=10):
    """
    주어진 이미지 경로의 테이블 영역에서 수평/수직 선을 검출하고,
    교차점 좌표 중 서로 threshold 픽셀 이내에 있는 좌표들을 그룹화하여
    필터링된 x, y 좌표 리스트를 반환하는 함수.
    """
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    # 그레이스케일 및 이진화 (adaptive threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, -2)
    
    # 수평 선 검출 (커널 크기는 이미지 폭의 일부)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 30, 1))
    h_lines_raw = cv2.erode(thresh, horizontal_kernel, iterations=1)
    h_lines_raw = cv2.dilate(h_lines_raw, horizontal_kernel, iterations=1)
    
    # 수직 선 검출 (커널 크기는 이미지 높이의 일부)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 30))
    v_lines_raw = cv2.erode(thresh, vertical_kernel, iterations=1)
    v_lines_raw = cv2.dilate(v_lines_raw, vertical_kernel, iterations=1)
    
    # 최소 길이 조건 적용
    # - 가로 선: 너비가 150픽셀 이상 (필요 시 값 조정)
    h_contours, _ = cv2.findContours(h_lines_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_lines = np.zeros_like(h_lines_raw)
    for cnt in h_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 150:
            cv2.drawContours(h_lines, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # - 세로 선: 높이가 100픽셀 이상
    v_contours, _ = cv2.findContours(v_lines_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_lines = np.zeros_like(v_lines_raw)
    for cnt in v_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= 100:
            cv2.drawContours(v_lines, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # 교차점(Intersection) 추출: 필터링된 수평선과 수직선의 교집합
    intersections = cv2.bitwise_and(h_lines, v_lines)
    
    # 교차점 좌표 추출
    contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    min_area = 10  # 너무 작은 잡음은 제외
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    
    # x좌표와 y좌표 리스트 분리
    x_list = [pt[0] for pt in points]
    y_list = [pt[1] for pt in points]
    
    # 서로 threshold 픽셀 이내에 있는 좌표 그룹화 (필터링)
    filtered_x = filter_close_coordinates(x_list, threshold=threshold)
    filtered_y = filter_close_coordinates(y_list, threshold=threshold)
    
    return filtered_x, filtered_y
