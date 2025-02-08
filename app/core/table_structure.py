# app/core/table_structure.py
import cv2
import numpy as np
from model.table import Table

def is_boundary_present(table_img, start_x, start_y, x_length, y_length, border_thickness=15):
    """
    셀의 경계가 존재하는지 확인하는 함수.
    Canny 엣지 검출 및 Hough 변환을 사용하여 상하좌우 선이 있는지 검사.
    """
    region = table_img[start_y-1:start_y+y_length+2, start_x-2:start_x+x_length+2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    def search_line(strip, length, min_ratio=0.7, hough_thresh=30, max_gap=2):
        min_length = int(length * min_ratio)
        lines = cv2.HoughLinesP(strip, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_length, maxLineGap=max_gap)
        return lines is not None and len(lines) > 0

    # 상, 하, 좌, 우 경계 검출
    top = 1 if search_line(edges[:border_thickness, :], x_length) else 0
    bottom = 1 if search_line(edges[-border_thickness:, :], x_length) else 0
    left = 1 if search_line(edges[:, :border_thickness], y_length) else 0
    right = 1 if search_line(edges[:, -border_thickness:], y_length) else 0

    # 4비트 플래그로 반환 (상3, 하2, 좌1, 우0)
    return (top << 3) | (bottom << 2) | (left << 1) | right

def check_merged_cells(table: Table, i, j, visited):
    """
    병합된 셀을 탐색하는 함수.
    상하좌우 이동하면서 연결된 셀들을 그룹으로 묶음.
    """
    stack, group = [(i, j)], []
    
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= table.n_rows or c < 0 or c >= table.n_cols or visited[r][c]:
            continue

        x_start, y_start = table.x_coords[c], table.y_coords[r]
        x_length, y_length = table.x_coords[c + 1] - x_start, table.y_coords[r + 1] - y_start
        flag = is_boundary_present(table.table_image, x_start, y_start, x_length, y_length)

        if flag == 15:  # 네 방향 모두 닫혀있으면 병합되지 않은 셀
            continue

        # 경계가 없는 방향으로 이동
        if not (flag >> 3) & 1: stack.append((r-1, c))  # 위
        if not (flag >> 2) & 1: stack.append((r+1, c))  # 아래
        if not (flag >> 1) & 1: stack.append((r, c-1))  # 왼쪽
        if not flag & 1: stack.append((r, c+1))        # 오른쪽

        visited[r][c] = True
        group.append((r, c))
    
    return group

def generate_table_structure(table: Table):
    """
    표 구조를 분석하여 개별 셀과 병합된 셀을 식별하고 테이블 객체에 저장.
    """
    visited = [[False] * table.n_cols for _ in range(table.n_rows)]
    
    for i in range(table.n_rows):
        for j in range(table.n_cols):
            if visited[i][j]:
                continue

            x_start, y_start = table.x_coords[j], table.y_coords[i]
            x_end, y_end = table.x_coords[j+1], table.y_coords[i+1]
            boundary = is_boundary_present(table.table_image, x_start, y_start, x_end - x_start, y_end - y_start)
            
            if boundary == 15:  # 개별 셀
                visited[i][j] = True
                cell = Table.Cell(i, j, x_start, y_start, x_end, y_end, None)
            else:  # 병합된 셀 탐색
                group = check_merged_cells(table, i, j, visited)
                r_list, c_list = zip(*group)
                x_start, y_start = table.x_coords[min(c_list)], table.y_coords[min(r_list)]
                x_end, y_end = table.x_coords[max(c_list)+1], table.y_coords[max(r_list)+1]
                cell = Table.Cell(i, j, x_start, y_start, x_end, y_end, None, is_merged=True, group=group)

            table.cells.append(cell)
            table.cell_matrix[i][j] = cell
    
    return table
