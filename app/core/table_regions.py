# app/core/table_regions.py
import os
import cv2

def extract_and_save_table_regions(image_path, output_prefix="table_region", min_area=3000, aspect_ratio_range=(0.5, 3.0)):
    """
    이미지에서 표 영역을 추출하여 개별 이미지 파일로 저장하는 함수.

    Parameters:
        image_path (str): 입력 이미지 파일 경로.
        output_prefix (str): 저장 파일 접두사 (기본값 "table_region").
        min_area (int): 표 후보 영역의 최소 면적 (기본값 3000).
        aspect_ratio_range (tuple): 표 영역의 가로세로 비율 범위 (기본값 (0.5, 3.0)).

    Returns:
        list: 추출된 표 영역 이미지의 저장 경로 리스트.
    """
    
    # 파일명에서 페이지 이름 추출 및 저장 디렉토리 생성
    base_filename = os.path.basename(image_path)
    page_name, _ = os.path.splitext(base_filename)
    output_dir = os.path.join("tables", page_name)
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    # 그레이스케일 변환 및 이진화
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 모폴로지 연산을 이용한 노이즈 제거 및 선 보정
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 외곽선 검출
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    saved_paths = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 외접 사각형 계산 및 가로세로 비율 검사
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
            table_img = image[y:y+h, x:x+w]
            
            # 개별 표 영역 저장
            output_filename = os.path.join(output_dir, f"{output_prefix}_{idx+1}.png")
            cv2.imwrite(output_filename, table_img)
            saved_paths.append(output_filename)
    
    return saved_paths
