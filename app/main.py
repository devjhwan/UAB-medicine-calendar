import os
from core import *
from helper import *
from model.table import Table

def process_page(page_image_path):
    """
    페이지 이미지를 처리하여 표를 추출하고 OCR을 수행.
    """
    messages = []
    page_num = int(os.path.splitext(os.path.basename(page_image_path))[0].split('_')[-1]) if '_' in page_image_path else 0
    print(f"[Page {page_num}] Processing started: {page_image_path}")

    # 표 영역 추출
    table_region_paths = extract_and_save_table_regions(page_image_path, output_prefix="table_region", min_area=3000, aspect_ratio_range=(0.5, 3.0))
    print(f"[Page {page_num}] Extracted {len(table_region_paths)} table regions")

    # 각 표 영역 처리
    for idx, table_region_path in enumerate(table_region_paths, start=1):
        print(f"[Page {page_num}, Table {idx}] Processing started")

        # 좌표 검출 및 테이블 구조 생성
        filtered_x, filtered_y = get_filtered_coordinates(table_region_path, threshold=10)
        table = generate_table_structure(Table(table_region_path, filtered_x, filtered_y, page_num, idx))

        # OCR 수행 및 JSON 저장
        extract_table_data(table)
        save_extracted_data_json(table, output_dir="extracted_ocr_data")

        print(f"[Page {page_num}, Table {idx}] Processing completed")
        messages.append(f"Page {page_num}, Table {idx} processed")

    return table, messages

def main_pipeline():
    """
    PDF를 이미지로 변환하고 각 페이지를 처리.
    """
    pdf_path = 'assets/DOC_HorariMed3_VH,0.pdf'
    page_image_paths = parse_pdf_to_images(pdf_path, image_dir='images', dpi_value=300)

    all_tables, all_messages = [], []
    for page_path in page_image_paths:
        table, msgs = process_page(page_path)
        for msg in msgs:
            print(msg)
        all_tables.append(table)
        all_messages.extend(msgs)

    # 최종 데이터 가공
    process_data_as_calendar_csv(all_tables)
    print("\n=== Processing Completed ===")

if __name__ == '__main__':
    main_pipeline()
