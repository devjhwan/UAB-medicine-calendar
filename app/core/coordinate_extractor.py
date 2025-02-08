# app/core/coordinate_extractor.py
import cv2
import numpy as np

def filter_close_coordinates(coords, threshold=10):
    """
    좌표 리스트에서 threshold 픽셀 이내에 가까운 값들을 그룹화하여
    각 그룹의 중앙값을 대표값으로 반환.
    """
    if not coords:
        return []
    
    coords.sort()
    filtered, group = [], [coords[0]]
    
    for coord in coords[1:]:
        if abs(coord - group[-1]) < threshold:
            group.append(coord)
        else:
            filtered.append(int(np.mean(group)))
            group = [coord]
    
    filtered.append(int(np.mean(group)))
    return filtered

def verify_line_histogram(gray, coord, orientation, crop_thickness=1, black_threshold=50, black_ratio_threshold=0.5):
    """
    이미지에서 지정된 좌표를 기준으로 픽셀의 검정색 비율을 분석하여
    유효한 테이블 선인지 확인.
    """
    start = max(coord - crop_thickness // 2, 0)
    
    if orientation == 'vertical':
        end = min(start + crop_thickness, gray.shape[1])
        crop = gray[:, start:end]
    elif orientation == 'horizontal':
        end = min(start + crop_thickness, gray.shape[0])
        crop = gray[start:end, :]
    else:
        raise ValueError("Invalid orientation. Use 'vertical' or 'horizontal'.")
    
    return np.count_nonzero(crop < black_threshold) / crop.size >= black_ratio_threshold

def get_filtered_coordinates(image_path, threshold=10, crop_thickness=1, black_threshold=50, black_ratio_threshold=0.5):
    """
    이미지에서 수평 및 수직선을 검출하고, 유효한 테이블 선 좌표를 필터링하여 반환.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # 이미지 전처리: 그레이스케일 변환 및 이진화
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)
    
    # 수평 및 수직선 검출을 위한 구조화 커널 생성
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 30))

    # 선 검출: 침식 후 팽창 (노이즈 제거)
    h_lines_raw = cv2.dilate(cv2.erode(thresh, horizontal_kernel, iterations=1), horizontal_kernel, iterations=1)
    v_lines_raw = cv2.dilate(cv2.erode(thresh, vertical_kernel, iterations=1), vertical_kernel, iterations=1)

    def filter_lines(raw_lines, min_length, axis):
        """
        검출된 선의 최소 길이를 기준으로 필터링하여 유효한 선만 남김.
        """
        contours, _ = cv2.findContours(raw_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_lines = np.zeros_like(raw_lines)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w if axis == 'horizontal' else h) >= min_length:
                cv2.drawContours(filtered_lines, [cnt], -1, 255, thickness=cv2.FILLED)
        return filtered_lines

    # 최소 길이 조건을 적용한 수평 및 수직선 필터링
    h_lines = filter_lines(h_lines_raw, 150, 'horizontal')
    v_lines = filter_lines(v_lines_raw, 150, 'vertical')

    # 수평선과 수직선의 교차점 추출
    intersections = cv2.bitwise_and(h_lines, v_lines)
    contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 교차점 좌표 계산 (중심 좌표 추출)
    points = [
        (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        for cnt in contours if cv2.contourArea(cnt) >= 10 and (M := cv2.moments(cnt))["m00"] != 0
    ]

    # 좌표 그룹화 및 필터링
    filtered_x = filter_close_coordinates([pt[0] for pt in points], threshold)
    filtered_y = filter_close_coordinates([pt[1] for pt in points], threshold)

    # 히스토그램 분석을 통해 유효한 선 좌표 검증
    valid_x = [x for x in filtered_x if verify_line_histogram(gray, x, 'vertical', crop_thickness, black_threshold, black_ratio_threshold)]
    valid_y = [y for y in filtered_y if verify_line_histogram(gray, y, 'horizontal', crop_thickness, black_threshold, black_ratio_threshold)]

    # 테이블 경계를 위한 보정 (최소 간격 유지)
    min_threshold, ylen, xlen = 30, img.shape[0], img.shape[1]
    
    if valid_x:
        if valid_x[0] > min_threshold: valid_x.insert(0, 2)
        if valid_x[-1] < xlen - min_threshold: valid_x.append(xlen - 2)
    
    if valid_y:
        if valid_y[0] > min_threshold: valid_y.insert(0, 2)
        if valid_y[-1] < ylen - min_threshold: valid_y.append(ylen - 2)

    return valid_x, valid_y
