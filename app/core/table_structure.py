# table_structure.py
import cv2
import numpy as np

def is_boundary_present(table_img, start_x, start_y, x_length, y_length, edge_thickness=1, intensity_threshold=50):
    """
    주어진 영역 ([start_x, start_x+x_length] × [start_y, start_y+y_length])의 상, 하, 좌, 우
    경계(각각 edge_thickness 픽셀 두께)의 평균 밝기를 측정하여, 
    밝기가 intensity_threshold 이하이면 해당 경계가 존재한다고 판단한다.
    
    각 경계의 존재 여부를 4개의 비트로 표현하여 정수로 반환한다.
    비트 순서는 [상, 하, 좌, 우]이며, 예:
      - 상: 0, 하: 1, 좌: 1, 우: 0 → 0110 (2진수) = 6 (10진수)
      - 상: 1, 하: 0, 좌: 0, 우: 1 → 1001 (2진수) = 9 (10진수)
    
    Parameters:
      - table_img: 원본 이미지 (컬러 이미지)
      - start_x, start_y, x_length, y_length: 관심 영역의 좌표와 크기
      - edge_thickness (int): 각 경계 검사에 사용할 픽셀 두께 (기본값 1)
      - intensity_threshold (int): 평균 밝기 임계값 (기본값 50)
    
    Returns:
      - boundary_flag (int): 0 ~ 15 사이의 정수
    """
    region = table_img[start_y:start_y+y_length, start_x:start_x+x_length]
    
    top_strip = region[0:edge_thickness, :]
    top_gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY)
    top_mean = np.mean(top_gray)
    
    bottom_strip = region[-edge_thickness:, :]
    bottom_gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
    bottom_mean = np.mean(bottom_gray)
    
    left_strip = region[:, 0:edge_thickness]
    left_gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY)
    left_mean = np.mean(left_gray)
    
    right_strip = region[:, -edge_thickness:]
    right_gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY)
    right_mean = np.mean(right_gray)
    
    top_flag    = 1 if top_mean <= intensity_threshold else 0
    bottom_flag = 1 if bottom_mean <= intensity_threshold else 0
    left_flag   = 1 if left_mean <= intensity_threshold else 0
    right_flag  = 1 if right_mean <= intensity_threshold else 0
    
    boundary_flag = (top_flag << 3) | (bottom_flag << 2) | (left_flag << 1) | right_flag
    return boundary_flag

def flood_fill_group(table_img, x_list, y_list, start_r, start_c, visited, n_rows, n_cols, intensity_threshold=50):
    """
    (start_r, start_c)에서 시작하여 4방향 flood fill 방식으로 그룹을 탐색한다.
    단, 현재 셀의 경계 플래그에서 열린(open) 방향(해당 bit가 0인 방향)으로만 이동한다.
    
    인접 셀 중 해당 셀의 boundary_flag가 15 (독립 셀)는 그룹에 포함하지 않고,
    boundary_flag가 15가 아닌 셀들을 동일 그룹으로 묶어, 그룹 내 셀의 인덱스 리스트를 반환한다.
    
    Parameters:
      - table_img: 원본 이미지 (컬러)
      - x_list, y_list: 기본 셀 경계를 나타내는 리스트 (길이: len(x_list)-1, len(y_list)-1 셀 수)
      - start_r, start_c: 시작 셀의 행, 열 인덱스
      - visited: 동일 크기의 2차원 boolean 리스트 (이미 방문한 셀 여부)
      - n_rows, n_cols: 기본 셀 그리드의 행, 열 수
      - intensity_threshold: 경계 판단 임계값
      
    Returns:
      - group: flood fill을 통해 탐색된 셀 인덱스들의 리스트 [(r, c), ...]
    """
    group = []
    stack = [(start_r, start_c)]
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
            continue
        if visited[r][c]:
            continue
        
        # 현재 셀의 경계 플래그 계산
        flag = is_boundary_present(table_img, x_list[c], y_list[r],
                                   x_list[c+1]-x_list[c], y_list[r+1]-y_list[r],
                                   edge_thickness=1, intensity_threshold=intensity_threshold)
        if flag == 15:
            # 경계가 모두 닫힌 독립 셀은 그룹에 포함하지 않음.
            continue
        
        visited[r][c] = True
        group.append((r, c))
        
        # 4방향 탐색: 단, 현재 셀의 경계 플래그에서 해당 방향이 '열린' 경우에만 이동
        # 상: dr = -1, dc = 0, bit 3 (값이 0이어야 함)
        if (flag >> 3) & 1 == 0 and r - 1 >= 0 and not visited[r-1][c]:
            stack.append((r-1, c))
        # 하: dr = +1, dc = 0, bit 2
        if (flag >> 2) & 1 == 0 and r + 1 < n_rows and not visited[r+1][c]:
            stack.append((r+1, c))
        # 좌: dr = 0, dc = -1, bit 1
        if (flag >> 1) & 1 == 0 and c - 1 >= 0 and not visited[r][c-1]:
            stack.append((r, c-1))
        # 우: dr = 0, dc = +1, bit 0
        if (flag >> 0) & 1 == 0 and c + 1 < n_cols and not visited[r][c+1]:
            stack.append((r, c+1))
    return group


def generate_table_structure(image_path, x_list, y_list, intensity_threshold=50):
    """
    주어진 원본 이미지와 x, y 좌표 리스트를 이용해 셀 그리드(행렬)를 구성하고,
    각 기본 셀에 대해 경계 검사(is_boundary_present)를 실시한다.
    
    독립 셀 (boundary_flag == 15)은 x_span, y_span = 1, merge_matrix 없음 (0)로 처리한다.
    병합이 필요한 셀은 flood fill 알고리즘으로 인접 셀 그룹을 탐색하고,  
    그룹의 최소 행, 최소 열을 기준으로 병합 영역 전체를 계산한 후,
    전체 병합 영역의 크기(x_span, y_span)만큼의 merge_matrix (2차원 리스트)를 생성한다.
    이 merge_matrix는 해당 그룹 내 모든 기본 셀 위치에 대해 그룹 인덱스를 할당한다.
    
    최종 반환:
      - cell_matrix: (len(y_list)-1) x (len(x_list)-1) 2차원 리스트. 각 활성 셀(병합 그룹의 좌측 상단 셀)은 셀 객체(dict),
        병합 영역 내 하위 셀은 None.
      - cells: 활성 셀 객체들의 리스트. 각 셀 객체는 'x_start', 'y_start', 'x_length', 'y_length',
        'x_span', 'y_span', 'data' 및 (병합 그룹인 경우) 'merge_matrix'는 포함하지 않고,
        전체 병합 그룹에 대한 정보는 별도의 merge_matrix가 반환됨.
      - merge_matrix: (len(y_list)-1) x (len(x_list)-1) 2차원 리스트.
        독립 셀의 경우 0, 병합 그룹에 속한 셀은 해당 그룹의 고유 인덱스(1부터 시작)로 표시된다.
    """
    table_img = cv2.imread(image_path)
    if table_img is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
        
    n_cols = len(x_list) - 1
    n_rows = len(y_list) - 1
    
    cell_matrix = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    merge_matrix = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    visited = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    cells = []
    current_group_index = 1  # 병합 그룹 인덱스는 1부터 시작
    
    for i in range(n_rows):
        for j in range(n_cols):
            if visited[i][j]:
                continue
            
            # 현재 기본 셀의 경계 플래그 계산
            flag = is_boundary_present(table_img, x_list[j], y_list[i],
                                       x_list[j+1]-x_list[j], y_list[i+1]-y_list[i],
                                       edge_thickness=1, intensity_threshold=intensity_threshold)
            if flag == 15:
                # 독립 셀
                cell = {
                    'x_start': x_list[j],
                    'y_start': y_list[i],
                    'x_length': x_list[j+1] - x_list[j],
                    'y_length': y_list[i+1] - y_list[i],
                    'x_span': 1,
                    'y_span': 1,
                    'group': 0,
                    'data': None
                }
                cells.append(cell)
                cell_matrix[i][j] = cell
                merge_matrix[i][j] = 0  # 독립 셀은 0으로 표시
                visited[i][j] = True
            else:
                # 병합 그룹: flood fill로 인접 셀들을 그룹화
                group = flood_fill_group(table_img, x_list, y_list, i, j, visited, n_rows, n_cols, intensity_threshold)
                if group:
                    # 그룹 내 최소, 최대 인덱스
                    rows = [r for (r, c) in group]
                    cols = [c for (r, c) in group]
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)
                    
                    # 병합 그룹 전체 영역
                    x_start = x_list[min_c]
                    y_start = y_list[min_r]
                    x_end = x_list[max_c+1]
                    y_end = y_list[max_r+1]
                    x_span = max_c - min_c + 1
                    y_span = max_r - min_r + 1
                    
                    # 활성 셀: 그룹의 좌측 상단 셀 (min_r, min_c) 에 대해 셀 객체 생성
                    cell = {
                        'x_start': x_start,
                        'y_start': y_start,
                        'x_length': x_end - x_start,
                        'y_length': y_end - y_start,
                        'x_span': x_span,
                        'y_span': y_span,
                        'group': current_group_index,
                        'data': None
                    }
                    cells.append(cell)
                    cell_matrix[min_r][min_c] = cell
                    # 그룹 내 모든 셀의 merge_matrix 값 설정 (merge_matrix 전체 그리드에 반영)
                    for (r, c) in group:
                        merge_matrix[r][c] = current_group_index
                    current_group_index += 1
    return cell_matrix, cells, merge_matrix

# 예시 사용 (테스트용)
if __name__ == '__main__':
    image_path = 'tables/page_1/table_region_20.png'
    # 예시 좌표 (실제 값은 표 추출 단계 결과에 따라 결정)
    x_list = [0, 50, 100, 150, 200]
    y_list = [0, 30, 60, 90, 120]
    cell_matrix, cells, merge_matrix = generate_table_structure(image_path, x_list, y_list)
    
    print("Merge Matrix:")
    for row in merge_matrix:
        print(row)
    
    print("\nActive Cells:")
    for cell in cells:
        print(cell)
