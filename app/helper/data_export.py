import os
import json
from model.table import Table

def save_extracted_data_json(table: Table, output_dir="extracted_ocr_data"):
    """
    Table 인스턴스의 cell_matrix에서 각 셀의 row, col, data, is_merged, merge_cols를
    추출하여 JSON 파일로 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"page{table.page_num}_table{table.table_idx}.json")
    
    cell_matrix_data = []
    for row in table.cell_matrix:
        row_data = []
        for cell in row:
            if cell is not None and cell.data != "":
                cell_data = {
                    "row": cell.row,
                    "col": cell.col,
                    "data": cell.data,
                    "is_merged": cell.is_merged
                }
            else:
                cell_data = None
            row_data.append(cell_data)
        cell_matrix_data.append(row_data)
    
    data = {"cell_matrix": cell_matrix_data}
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted OCR data saved: {filename}")