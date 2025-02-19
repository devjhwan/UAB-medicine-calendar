# debug.py
import os
import cv2
from core.ocr_processing import preprocess_cell_image
from model.table import Table

def draw_cell_boundaries(table: Table, output_dir="debug_boundaries"):
    """
    주어진 Table 객체를 이용하여,  
    각 활성 셀(독립 셀 또는 병합 셀의 좌측 상단 셀)에 대해 해당 셀 영역을  
    빨간색 사각형과 병합 정보를 나타내는 텍스트(예: "2x1" 또는 "1x1")로 오버레이한  
    디버깅용 이미지를 생성하여 저장한다.
    
    저장 경로는 "debug_boundaries/page{page_num}_table{table_idx}.png" 형식이다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 원본 이미지 복사 (원본 보존)
    image = table.table_image.copy()
    
    # Table.cell_matrix의 각 활성 셀에 대해 경계 오버레이
    for row in table.cell_matrix:
        for cell in row:
            if cell is None:
                continue
            
            x_start = cell.x_start
            y_start = cell.y_start
            x_end = cell.x_end
            y_end = cell.y_end
            # 병합 셀인지 독립 셀인지 판단:
            if cell.is_merged:
                # cell.area: ((min_r, max_r), (min_c, max_c))
                (min_r, max_r), (min_c, max_c) = cell.area
                # 병합 셀의 크기
                text = f"{max_c - min_c + 1}x{max_r - min_r + 1}"
            else:
                # 독립 셀의 크기
                text = "1x1"
            
            # 사각형 오버레이 (빨간색, 두께 2)
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            # 텍스트 오버레이 (좌측 상단)
            cv2.putText(image, text, (x_start + 5, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    output_filename = os.path.join(output_dir, f"page{table.page_num}_table{table.table_idx}.png")
    cv2.imwrite(output_filename, image)
    print(f"디버깅용 셀 경계 이미지 저장됨: {output_filename}")
    
def draw_grid_on_image(table: Table, output_dir="debug_layouts"):
    """
    주어진 원본 이미지와 필터링된 x, y 좌표 리스트를 이용하여,
    얇은 빨간 선(두께 1)으로 그리드를 원본 이미지에 덮어씌운 후,
    "debug_layouts/page{page_num}table{table_num}.png" 형식으로 저장하는 함수.
    
    Parameters:
      - image_path (str): 원본 이미지 파일 경로.
      - filtered_x (list): 필터링된 x 좌표 리스트.
      - filtered_y (list): 필터링된 y 좌표 리스트.
      - page_num (int): 페이지 번호.
      - table_num (int): 테이블 번호.
      - output_dir (str): 디버깅 이미지를 저장할 기본 디렉토리 (기본값 "debug_layouts").
    """
    # 원본 이미지 읽기
    image = table.table_image
    
    # 수직선 그리기: 각 x 좌표에 대해 이미지의 상단부터 하단까지 빨간 선 그리기 (두께 1)
    for x in table.x_coords:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 0, 255), 1)
    
    # 수평선 그리기: 각 y 좌표에 대해 이미지의 좌측부터 우측까지 빨간 선 그리기 (두께 1)
    for y in table.y_coords:
        cv2.line(image, (0, y), (image.shape[1], y), (0, 0, 255), 1)
    
    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"page{table.page_num}table{table.table_idx}.png")
    cv2.imwrite(output_filename, image)
    print(f"Debug layout image saved: {output_filename}")
    