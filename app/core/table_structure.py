# table_structure.py
import cv2
import numpy as np
from model.table import Table

def is_boundary_present(table_img, start_x, start_y, x_length, y_length,
                        border_thickness=15):
    # 관심 영역 추출
    region = table_img[start_y-1:start_y+y_length+2, start_x-2:start_x+x_length+2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    def search_line(strip, length, min_line_length_ratio=0.7,
                    hough_threshold=30, max_line_gap=2):
        min_line_length = int(length * min_line_length_ratio)
        
        lines = cv2.HoughLinesP(strip, 1, np.pi/180,
                                threshold=hough_threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        return lines is not None and len(lines) > 0
    
    top_strip = edges[:border_thickness, :]
    bottom_strip = edges[-border_thickness:, :]
    left_strip = edges[:, :border_thickness]
    right_strip = edges[:, -border_thickness:]
    
    top_flag    = 1 if search_line(top_strip, x_length) else 0
    bottom_flag = 1 if search_line(bottom_strip, x_length)else 0
    left_flag   = 1 if search_line(left_strip, y_length) else 0
    right_flag  = 1 if search_line(right_strip, y_length) else 0
    
    # 비트 순서: 상, 하, 좌, 우
    boundary_flag = (top_flag << 3) | (bottom_flag << 2) | (left_flag << 1) | right_flag
    return boundary_flag

def check_merged_cells(table: Table, i, j, visited):
    stack = [(i, j)]
    group = []
    
    while stack:
        r, c = stack.pop()
        if (r < 0 or r >= table.n_rows) or \
           (c < 0 or c >= table.n_cols) or \
           (visited[r][c]):
            continue
        x_start = table.x_coords[c]
        y_start = table.y_coords[r]
        x_length = table.x_coords[c + 1] - table.x_coords[c]
        y_length = table.y_coords[r + 1] - table.y_coords[r]
        
        flag = is_boundary_present(table.table_image, x_start, y_start, x_length, y_length)
        
        if flag == 15:
            continue
        # 4방향 탐색: 현재 셀의 경계 플래그에서 해당 방향이 열린 경우에만 이동.
        # 상: bit 3, 하: bit 2, 좌: bit 1, 우: bit 0.
        if ((flag >> 3) & 1) == 0:
            stack.append((r-1, c))
        if ((flag >> 2) & 1) == 0:
            stack.append((r+1, c))
        if ((flag >> 1) & 1) == 0:
            stack.append((r, c-1))
        if ((flag >> 0) & 1) == 0:
            stack.append((r, c+1))
        visited[r][c] = True
        group.append((r, c))
    return group

def group_consecutive_pairs(pairs):
    """
    (row, col) 쌍 리스트를 정렬한 후, 같은 행에서 연속하는 열을 그룹화하여
    (row, (start_col, end_col)) 형태의 리스트로 반환합니다.
    
    예)
        Input: [(2,1),(2,2),(2,3),(2,5),(2,6),(3,2),(3,3)]
        Output: [(2, (1, 3)), (2, (5, 6)), (3, (2, 3))]
    """
    # 첫 번째 원소(행) 기준, 그 다음 열 기준으로 정렬
    pairs_sorted = sorted(pairs, key=lambda x: (x[0], x[1]))
    
    groups = []
    current_group = []
    
    for pair in pairs_sorted:
        if not current_group:
            current_group.append(pair)
        else:
            last = current_group[-1]
            # 같은 행이고 열이 연속이면 그룹에 추가
            if pair[0] == last[0] and pair[1] == last[1] + 1:
                current_group.append(pair)
            else:
                # 현재 그룹 마무리: (row, (start, end)) 형식으로 저장
                row = current_group[0][0]
                start_col = current_group[0][1]
                end_col = current_group[-1][1]
                groups.append((row, (start_col, end_col)))
                # 새 그룹 시작
                current_group = [pair]
    # 마지막 그룹 처리
    if current_group:
        row = current_group[0][0]
        start_col = current_group[0][1]
        end_col = current_group[-1][1]
        groups.append((row, (start_col, end_col)))
    
    return groups
        
def generate_table_structure(table: Table):
    visited = [[False for _ in range(table.n_cols)] for _ in range(table.n_rows)]
    for i in range(table.n_rows):
        for j in range(table.n_cols):
            if visited[i][j]:
                continue
            x_start = table.x_coords[j]
            y_start = table.y_coords[i]
            x_length = table.x_coords[j + 1] - table.x_coords[j]
            y_length = table.y_coords[i + 1] - table.y_coords[i]
            
            boundary = is_boundary_present(table.table_image, x_start, y_start, x_length, y_length)
            if boundary == 15:
                #Single Cell
                visited[i][j] = True
                cell = Table.Cell(i, j, x_start, y_start, table.x_coords[j+1], table.y_coords[i+1], None)
            else:
                #Merged Cell
                group = check_merged_cells(table, i, j, visited)
                r_list, c_list = map(list, zip(*group))
                min_r, max_r = np.min(r_list), np.max(r_list)
                min_c, max_c = np.min(c_list), np.max(c_list)
                
                x_start = table.x_coords[min_c]
                y_start = table.y_coords[min_r]
                x_end = table.x_coords[max_c+1]
                y_end = table.y_coords[max_r+1]
                
                area = ((min_r, max_r), (min_c, max_c))
                compressed_group = group_consecutive_pairs(group)
                cell = Table.Cell(i, j, x_start, y_start, x_end, y_end, None, \
                                    is_merged=True, area=area, merge_cols=compressed_group)
            table.cells.append(cell)
            table.cell_matrix[i][j] = cell
    return table
