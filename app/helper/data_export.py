# data_export.py
import os
import json

def save_extracted_data_json(page_num, table_num, cell_matrix, cells, output_dir="extracted_ocr_data"):
    """
    주어진 cell_matrix와 cells 정보를 JSON 형식으로 저장한다.
    저장 파일명은 "extracted_ocr_data/page_{page_num}table{table_num}.json"이다.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"page{page_num}_table{table_num}.json")
    
    # JSON 직렬화 시, numpy 데이터는 기본적으로 직렬화되지 않으므로,
    # cell_matrix와 cells 내의 모든 값들이 기본 타입(int, float, str 등)인지 확인한다.
    # (cell_matrix와 cells는 이전 단계에서 일반 dict와 리스트로 구성되므로 문제가 없을 것으로 가정)
    data = {
        "cell_matrix": cell_matrix,
        "cells": cells
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted OCR data saved: {filename}")
