import cv2
import numpy as np

def filter_close_coordinates(coords, threshold=10):
    """
    주어진 좌표 리스트에서 서로 threshold 픽셀 이내에 가까운 값들을 그룹화하여,
    각 그룹의 중앙값을 대표값으로 반환합니다.
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

def verify_line_histogram(gray, coord, orientation, crop_thickness=1,
                          black_threshold=50, black_ratio_threshold=0.5):
    """
    회색조 이미지에서 지정된 좌표를 기준으로 crop한 영역의 히스토그램을 분석합니다.
    crop_thickness 픽셀 두께의 영역에서 픽셀 값이 black_threshold 미만인 픽셀 비율이
    black_ratio_threshold 이상이면 경계선으로 판단하여 True를 반환합니다.
    """
    if orientation == 'vertical':
        start = max(coord - crop_thickness // 2, 0)
        end = min(start + crop_thickness, gray.shape[1])
        crop = gray[:, start:end]
    elif orientation == 'horizontal':
        start = max(coord - crop_thickness // 2, 0)
        end = min(start + crop_thickness, gray.shape[0])
        crop = gray[start:end, :]
    else:
        raise ValueError("Invalid orientation. Use 'vertical' or 'horizontal'.")
    
    black_pixels = np.count_nonzero(crop < black_threshold)
    total_pixels = crop.size
    ratio = black_pixels / total_pixels
    return ratio >= black_ratio_threshold

def get_filtered_coordinates(image_path, threshold=10,
                             crop_thickness=1, black_threshold=50, black_ratio_threshold=0.5):
    """
    주어진 이미지 경로의 테이블 영역에서 수평/수직 선을 검출하고,
    교차점 좌표 중 서로 threshold 픽셀 이내에 있는 좌표들을 그룹화하여
    필터링된 x, y 좌표 리스트를 반환합니다.
    
    추가로, 각 선 좌표에 대해 1px 두께의 영역 내에서
    검정색 픽셀 비율이 black_ratio_threshold 이상이면 유효한 경계선으로 판단합니다.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, -2)
    
    # 수평선 검출
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 30, 1))
    h_lines_raw = cv2.erode(thresh, horizontal_kernel, iterations=1)
    h_lines_raw = cv2.dilate(h_lines_raw, horizontal_kernel, iterations=1)
    
    # 수직선 검출
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 30))
    v_lines_raw = cv2.erode(thresh, vertical_kernel, iterations=1)
    v_lines_raw = cv2.dilate(v_lines_raw, vertical_kernel, iterations=1)
    
    # 최소 길이 조건 적용 (가로 선: 너비 150px 이상)
    h_contours, _ = cv2.findContours(h_lines_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_lines = np.zeros_like(h_lines_raw)
    for cnt in h_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 150:
            cv2.drawContours(h_lines, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # 최소 길이 조건 적용 (세로 선: 높이 150px 이상)
    v_contours, _ = cv2.findContours(v_lines_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_lines = np.zeros_like(v_lines_raw)
    for cnt in v_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= 150:
            cv2.drawContours(v_lines, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # 교차점 추출 (수평선과 수직선의 교집합)
    intersections = cv2.bitwise_and(h_lines, v_lines)
    contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    min_area = 10
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    
    x_list = [pt[0] for pt in points]
    y_list = [pt[1] for pt in points]
    filtered_x = filter_close_coordinates(x_list, threshold=threshold)
    filtered_y = filter_close_coordinates(y_list, threshold=threshold)
    
    # 히스토그램 분석을 통한 추가 검증
    valid_filtered_x = [x for x in filtered_x
                        if verify_line_histogram(gray, x, 'vertical',
                                                   crop_thickness, black_threshold, black_ratio_threshold)]
    valid_filtered_y = [y for y in filtered_y
                        if verify_line_histogram(gray, y, 'horizontal',
                                                   crop_thickness, black_threshold, black_ratio_threshold)]
    
    min_threshold = 30
    ylen, xlen = img.shape[:2]
    if valid_filtered_x[0] > min_threshold:
        valid_filtered_x.insert(0, 2)
    if valid_filtered_x[-1] < xlen - min_threshold:
        valid_filtered_x.append(xlen - 2)
    if valid_filtered_y[0] > min_threshold:
        valid_filtered_y.insert(0, 2)
    if valid_filtered_y[-1] < ylen - min_threshold:
        valid_filtered_y.append(ylen - 2)

    return valid_filtered_x, valid_filtered_y
