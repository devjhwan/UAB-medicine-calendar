import cv2
import numpy as np
import os

def extract_and_save_table_regions(image_path, output_prefix="table_region", min_area=3000, aspect_ratio_range=(0.5, 3.0)):
    """
    주어진 이미지에서 표 영역을 추출하여 개별 이미지 파일로 저장한다.
    저장 디렉토리는 image_path에서 추출한 파일 이름(확장자 제외)을 사용하여 "tables/page_n" 형식으로 생성된다.
    
    Parameters:
        image_path (str): 입력 이미지 파일 경로. 예를 들어 ".../page_1.png" 형태.
        output_prefix (str): 저장되는 파일명의 접두사 (기본값 "table_region").
        min_area (int): 표 후보 영역의 최소 면적 (픽셀 단위, 기본값 3000).
        aspect_ratio_range (tuple): 표 영역의 가로세로 비율 범위 (min_ratio, max_ratio), 기본값 (0.5, 3.0).
    """
    # image_path에서 파일명(확장자 제외) 추출 (예: "page_1")
    base_filename = os.path.basename(image_path)
    page_name, _ = os.path.splitext(base_filename)
    
    # 저장 디렉토리 설정: "tables/page_n" 형식
    output_dir = os.path.join("tables", page_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    orig_image = image.copy()
    
    # 2. 전처리: 그레이스케일 변환 및 이진화
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Morphological 연산: 팽창과 침식(CLOSING)으로 선 보정
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. 컨투어 검출
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 표 영역 후보 추출 및 필터링 후 저장
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 외접 사각형 구하기
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
            # 표 영역 추출
            table_img = image[y:y+h, x:x+w]
            # 출력 파일 경로 구성: output_dir 내에 output_prefix_번호.png로 저장
            output_filename = os.path.join(output_dir, f"{output_prefix}_{idx+1}.png")
            cv2.imwrite(output_filename, table_img)
            # (옵션) 원본 이미지에 검출 영역 표시 (디버그용)
            cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # (옵션) 디버깅: 검출 영역이 표시된 원본 이미지 저장 또는 출력
    # cv2.imwrite(os.path.join(output_dir, "detected_tables_debug.png"), orig_image)
    # cv2.imshow("Detected Tables", orig_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # 예시: 이미지 파일 경로가 "page_1.png", "page_2.png" 등으로 저장되어 있다고 가정
    image_path = 'images/page_3.png'
    extract_and_save_table_regions(image_path)
