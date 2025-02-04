# main.py
import os
from core import parse_pdf_to_images, extract_and_save_table_regions, \
                 get_filtered_coordinates, extract_table_data
from helper import save_preprocessed_merged_cells, draw_cell_boundaries, \
                   draw_grid_on_image, save_extracted_data_json
from model.table import Table

def process_page(page_image_path):
    """
    한 페이지의 이미지(page_image_path)를 받아서,
      - 해당 페이지에서 표 영역 이미지들을 생성 (table_regions)
      - 각 표 영역에 대해 좌표 추출 → 셀 그리드 및 병합 셀 구성 → OCR 데이터 추출
      - 각 테이블(표 영역)에 대해 결과를 JSON 파일로 저장
    각 테이블 작업이 완료되면 "Page {n}, Table {m} processed" 메시지를 생성하여 반환한다.
    """
    messages = []
    
    # 페이지 번호 추출 (예: "page_1.png" → 1)
    base_page = os.path.basename(page_image_path)
    page_name, _ = os.path.splitext(base_page)
    try:
        page_num = int(page_name.split('_')[1])
    except Exception as e:
        page_num = 0
    print(f"[Page Processing] Starting processing for page {page_num} (Image: {page_image_path})")
    
    # 2. 표 영역 이미지 추출
    print(f"[Table Regions] Starting table regions extraction on page {page_num}")
    table_region_paths = extract_and_save_table_regions(page_image_path,
                                                        output_prefix="table_region",
                                                        min_area=3000,
                                                        aspect_ratio_range=(0.5, 3.0))
    print(f"[Table Regions] Extracted {len(table_region_paths)} table region images on page {page_num}")
    
    # 각 테이블 영역 별 처리
    for table_idx, table_region_path in enumerate(table_region_paths[0:1], start=1):
        print(f"[Page {page_num}, Table {table_idx}] Starting processing")
        # 좌표 추출
        filtered_x, filtered_y = get_filtered_coordinates(table_region_path, threshold=10)
        print(f"[Page {page_num}, Table {table_idx}] Filtered coordinates extracted: x: {filtered_x}, y: {filtered_y}")
        
        # draw_grid_on_image(table_region_path, filtered_x, filtered_y, page_num, table_idx)
        
        # 셀 그리드 및 병합 셀 구성
        table = Table.from_image(table_region_path, filtered_x, filtered_y)
        print(f"[Page {page_num}, Table {table_idx}] Cell grid generated")
        
        draw_cell_boundaries(table, page_num, table_idx)
        
        # # OCR 데이터 추출
        # extract_table_data(table_region_path, cells)
        # print(f"[Page {page_num}, Table {table_idx}] OCR data extraction completed")
        
        # # JSON 파일에 추출된 결과 저장
        # save_extracted_data_json(page_num, table_idx, cell_matrix, cells, output_dir="extracted_ocr_data")
        # print(f"[Page {page_num}, Table {table_idx}] JSON data saved")
        
        # (옵션) 디버깅: 전처리된 셀 이미지 저장
        # save_preprocessed_merged_cells(table_region_path, cell_matrix, output_base_dir="preprocessed")
        
        messages.append(f"Page {page_num}, Table {table_idx} processed")
    
    return messages

def main_pipeline():
    print("Starting PDF to image conversion...")
    pdf_path = 'assets/DOC_HorariMed3_VH,0.pdf'
    page_image_paths = parse_pdf_to_images(pdf_path, image_dir='images', dpi_value=300)
    print(f"PDF to image conversion completed: {len(page_image_paths)} pages extracted\n")
    
    all_messages = []
    # 순차적으로 각 페이지 처리
    for page_path in page_image_paths:
        msgs = process_page(page_path)
        for msg in msgs:
            print(msg)
        all_messages.extend(msgs)
    
    print("\n=== 모든 페이지 및 테이블 작업 완료 ===")

if __name__ == '__main__':
    main_pipeline()
