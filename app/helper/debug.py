# debug.py
import os
import cv2
from core.ocr_processing import preprocess_cell_image

def save_preprocessed_merged_cells(image_path, cell_matrix, output_base_dir="preprocessed"):
    """
    주어진 원본 이미지와 셀 행렬(cell_matrix)을 이용해 각 활성 셀(병합 셀의 좌측 상단 셀)에 해당하는
    전체 병합 영역을 크롭한 후, 전처리(preprocess_cell_image)를 적용하여 디버깅용 이미지로 저장한다.
    
    저장 경로는 "preprocessed/{row}/{col}.png" 형식이며, None인 (병합 영역 내 하위 셀) 부분은 건너뛴다.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    for i, row in enumerate(cell_matrix):
        for j, cell in enumerate(row):
            if cell is None:
                continue
            x_start = cell['x_start']
            y_start = cell['y_start']
            x_length = cell['x_length']
            y_length = cell['y_length']
            cropped_cell = original_img[y_start:y_start+y_length, x_start:x_start+x_length]
            preprocessed_img = preprocess_cell_image(cropped_cell, crop_px=4, rescale_factor=2)
            
            row_dir = os.path.join(output_base_dir, f"{i}")
            os.makedirs(row_dir, exist_ok=True)
            output_path = os.path.join(row_dir, f"{j}.png")
            cv2.imwrite(output_path, preprocessed_img)
            print(f"Saved preprocessed cell at row {i}, col {j}: {output_path}")

# app/helper/debug.py
import os
import cv2

def draw_cell_boundaries(table, page_num, table_num, output_dir="debug_boundaries"):
    """
    주어진 Table 객체를 이용하여, 각 셀에 대해 셀 경계를 원본 이미지에 오버레이한 디버깅용 이미지를 생성하여 저장한다.
    
    - Table 객체는 다음 속성을 가진다:
         x_coords: x 좌표 리스트 (길이: n+1)
         y_coords: y 좌표 리스트 (길이: m+1)
         cell_matrix: (m x n) 2차원 리스트, 각 원소는 독립 셀이면 {'row': i, 'col': j, 'data': None},
                      병합 셀의 활성 셀이면 {'row': i, 'col': j, 'row-area': (min_r, max_r), 'col-area': (min_c, max_c), 'data': None}
    - 독립 셀의 경우, 해당 cell의 픽셀 영역은
         x_start = x_coords[cell['col']], y_start = y_coords[cell['row']]
         x_end = x_coords[cell['col']+1], y_end = y_coords[cell['row']+1]
    - 병합 셀(활성 셀)의 경우, 픽셀 영역은
         x_start = x_coords[min_c], y_start = y_coords[min_r]
         x_end = x_coords[max_c+1], y_end = y_coords[max_r+1]
         여기서 cell['row-area'] = (min_r, max_r), cell['col-area'] = (min_c, max_c)
    
    생성된 디버깅 이미지는 "debug_boundaries/page{page_num}_table{table_num}.png"로 저장된다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 원본 이미지 읽기
    image = cv2.imread(table.table_image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + table.table_image_path)
    
    # 각 셀에 대해 그리드 경계 오버레이
    for row in table.cell_matrix:
        for cell in row:
            if cell is None:
                continue
            
            # 독립 셀인지 병합 활성 셀인지 판단
            if 'row-area' in cell and 'col-area' in cell:
                # 병합 활성 셀: 병합 그룹의 시작 셀
                min_r, max_r = cell['row-area']
                min_c, max_c = cell['col-area']
                x_start = table.x_coords[min_c]
                y_start = table.y_coords[min_r]
                x_end = table.x_coords[max_c+1]
                y_end = table.y_coords[max_r+1]
                text = f"{max_c-min_c+1}x{max_r-min_r+1}"
            else:
                # 독립 셀
                x_start = table.x_coords[cell['col']]
                y_start = table.y_coords[cell['row']]
                x_end = table.x_coords[cell['col']+1]
                y_end = table.y_coords[cell['row']+1]
                text = "1x1"
            
            # 사각형 그리기: 빨간색, 두께 2
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            # 텍스트 오버레이 (좌측 상단에)
            cv2.putText(image, text, (x_start + 5, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    output_filename = os.path.join(output_dir, f"page{page_num}_table{table_num}.png")
    cv2.imwrite(output_filename, image)
    print(f"디버깅용 셀 경계 이미지 저장됨: {output_filename}")

    
def draw_grid_on_image(image_path, filtered_x, filtered_y, page_num, table_num, output_dir="debug_layouts"):
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
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    # 수직선 그리기: 각 x 좌표에 대해 이미지의 상단부터 하단까지 빨간 선 그리기 (두께 1)
    for x in filtered_x:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 0, 255), 1)
    
    # 수평선 그리기: 각 y 좌표에 대해 이미지의 좌측부터 우측까지 빨간 선 그리기 (두께 1)
    for y in filtered_y:
        cv2.line(image, (0, y), (image.shape[1], y), (0, 0, 255), 1)
    
    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"page{page_num}table{table_num}.png")
    cv2.imwrite(output_filename, image)
    print(f"Debug layout image saved: {output_filename}")
    