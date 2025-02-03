# table_structure.py
import cv2
import numpy as np

def is_vertical_boundary_present(table_img, start_x, start_y, x_length, y_length,
                                 edge_width=1, intensity_threshold=50):
    """
    주어진 영역([start_x, start_x+x_length]×[start_y, start_y+y_length])의 
    오른쪽 edge_width 픽셀 영역의 평균 색상이 intensity_threshold 이하이면 
    경계선(어두운 선)이 존재한다고 판단한다.
    
    Parameters:
      - table_img: 원본 이미지
      - start_x, start_y, x_length, y_length: 관심 영역의 위치와 크기
      - edge_width (int): 검사할 우측 영역의 폭 (기본값 1)
      - intensity_threshold (int): 평균 픽셀 강도 임계값 (기본값 50)
      
    Returns:
      - True if the average intensity of the rightmost edge is <= intensity_threshold, else False.
    """
    region = table_img[start_y:start_y+y_length, start_x:start_x+x_length]
    # 가장 우측 edge_width 영역 추출 (모든 행, 마지막 edge_width 열)
    vertical_strip = region[:, -edge_width:]
    gray_strip = cv2.cvtColor(vertical_strip, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_strip)
    return mean_intensity <= intensity_threshold

def is_horizontal_boundary_present(table_img, start_x, start_y, x_length, y_length,
                                   edge_height=1, intensity_threshold=50):
    """
    주어진 영역([start_x, start_x+x_length]×[start_y, start_y+y_length])의 
    아래쪽 edge_height 영역의 평균 색상이 intensity_threshold 이하이면 
    경계선(어두운 선)이 존재한다고 판단한다.
    
    Parameters:
      - table_img: 원본 이미지
      - start_x, start_y, x_length, y_length: 관심 영역의 위치와 크기
      - edge_height (int): 검사할 아래쪽 영역의 높이 (기본값 1)
      - intensity_threshold (int): 평균 픽셀 강도 임계값 (기본값 50)
      
    Returns:
      - True if the average intensity of the bottom edge is <= intensity_threshold, else False.
    """
    region = table_img[start_y:start_y+y_length, start_x:start_x+x_length]
    # 가장 아래 edge_height 영역 추출 (모든 열, 마지막 edge_height 행)
    horizontal_strip = region[-edge_height:, :]
    gray_strip = cv2.cvtColor(horizontal_strip, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_strip)
    return mean_intensity <= intensity_threshold

def generate_table_structure(image_path, x_list, y_list):
    """
    주어진 원본 이미지와 x, y 좌표 리스트를 이용해 셀 그리드(행렬)를 구성하고,
    현재 셀부터 시작하여 우측 및 아래쪽 경계선을 검사해 병합 셀을 판별한다.
    
    출력:
      - cell_matrix: (len(y_list)-1) x (len(x_list)-1) 2차원 리스트.
                     각 활성 셀(병합 셀의 좌측 상단 셀)은 셀 객체(dict)로 저장,
                     병합 영역 내 하위 셀은 None.
      - cells: 활성 셀 객체들의 리스트.
    """
    table_img = cv2.imread(image_path)
    if table_img is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    n_cols = len(x_list) - 1
    n_rows = len(y_list) - 1
    cell_matrix = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    visited = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    cells = []

    for i in range(n_rows):
        for j in range(n_cols):
            if visited[i][j]:
                continue

            x_start = x_list[j]
            y_start = y_list[i]
            x_span = 1
            y_span = 1

            # 우측 병합 검사 (현재 셀부터 시작)
            if not is_vertical_boundary_present(table_img, x_list[j], y_start,
                                                x_list[j+1]-x_list[j],
                                                y_list[i+1]-y_start):
                k = j + 1
                while k < n_cols:
                    if is_vertical_boundary_present(table_img, x_list[k],
                                                    y_start,
                                                    x_list[k+1]-x_list[k],
                                                    y_list[i+1]-y_start):
                        k += 1  # 경계선이 발견된 셀까지 포함
                        break
                    else:
                        k += 1
                x_span = k - j
            else:
                x_span = 1

            # 아래쪽 병합 검사 (현재 셀부터 시작)
            if not is_horizontal_boundary_present(table_img, x_list[j],
                                                  y_list[i],
                                                  x_list[j+1]-x_list[j],
                                                  y_list[i+1]-y_list[i]):
                r = i + 1
                while r < n_rows:
                    can_merge = True
                    for col in range(j, j + x_span):
                        if is_horizontal_boundary_present(table_img, x_list[col],
                                                          y_list[r],
                                                          x_list[col+1]-x_list[col],
                                                          y_list[r+1]-y_list[r]):
                            r += 1  # 경계선이 발견된 행까지 포함
                            can_merge = False
                            break
                    if not can_merge:
                        break
                    else:
                        r += 1
                y_span = r - i
            else:
                y_span = 1

            x_end = x_list[j + x_span]
            y_end = y_list[i + y_span]
            for r_idx in range(i, i + y_span):
                for c_idx in range(j, j + x_span):
                    visited[r_idx][c_idx] = True
            cell = {
                'x_start': x_start,
                'y_start': y_start,
                'x_length': x_end - x_start,
                'y_length': y_end - y_start,
                'x_span': x_span,
                'y_span': y_span,
                'data': None
            }
            cells.append(cell)
            cell_matrix[i][j] = cell

    return cell_matrix, cells
