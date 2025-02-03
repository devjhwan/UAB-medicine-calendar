import cv2
import numpy as np
import os

def is_vertical_boundary_present(table_img, start_x, start_y, x_length, y_length,
                                 edge_width=5, min_line_length_ratio=0.7,
                                 canny_thresh1=50, canny_thresh2=150,
                                 hough_threshold=30, max_line_gap=5):
    """
    주어진 영역([start_x, start_x+x_length]×[start_y, start_y+y_length])의 오른쪽
    부분(폭 edge_width)에서 세로 경계선이 존재하는지 판단.
    최소 선 길이는 y_length * min_line_length_ratio 이상이어야 함.
    """
    region = table_img[start_y:start_y+y_length, start_x:start_x+x_length]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    vertical_strip = edges[:, -edge_width:]
    min_line_length = int(y_length * min_line_length_ratio)
    lines = cv2.HoughLinesP(vertical_strip, 1, np.pi/180,
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines is not None and len(lines) > 0

def is_horizontal_boundary_present(table_img, start_x, start_y, x_length, y_length,
                                   edge_height=5, min_line_length_ratio=0.7,
                                   canny_thresh1=50, canny_thresh2=150,
                                   hough_threshold=30, max_line_gap=5):
    """
    주어진 영역([start_x, start_x+x_length]×[start_y, start_y+y_length])의 아래쪽
    부분(높이 edge_height)에서 가로 경계선이 존재하는지 판단.
    최소 선 길이는 x_length * min_line_length_ratio 이상이어야 함.
    """
    region = table_img[start_y:start_y+y_length, start_x:start_x+x_length]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    horizontal_strip = edges[-edge_height:, :]
    min_line_length = int(x_length * min_line_length_ratio)
    lines = cv2.HoughLinesP(horizontal_strip, 1, np.pi/180,
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines is not None and len(lines) > 0

def generate_table_structure(image_path, x_list, y_list):
    """
    주어진 원본 이미지 경로와 x, y 좌표 리스트를 이용해 기본 그리드(행렬)을 구성하고,
    현재 셀부터 우측 및 아래쪽 경계선을 검사하여 병합 셀을 판별한다.
    
    입력:
      - image_path (str): 원본 테이블 이미지 경로.
      - x_list (list): 열 경계 리스트 (픽셀 단위). 예: [0, 50, 100, 150, 200]
      - y_list (list): 행 경계 리스트 (픽셀 단위). 예: [0, 30, 60, 90, 120]  
        (마지막 값은 표의 하단 경계)
    
    출력:
      - cell_matrix: 크기는 (len(y_list)-1) x (len(x_list)-1) 2차원 리스트.  
                     병합 셀의 좌측 상단 셀에는 셀 객체(dict)를 저장하고,  
                     병합 영역의 나머지 셀은 None.
      - cells: 활성 셀(병합 셀의 첫 셀) 객체들의 리스트.
    
    각 셀 객체(dict)는 다음 정보를 포함:
      - 'x_start', 'y_start': 시작 좌표
      - 'x_length', 'y_length': 전체 셀 영역의 크기
      - 'x_span', 'y_span': 병합된 셀의 칸 수
      - 'data': 빈 데이터 변수 (추후 사용)
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
                continue  # 이미 병합 영역에 포함된 셀은 건너뜀

            # 시작 셀의 기본 영역
            x_start = x_list[j]
            y_start = y_list[i]
            x_span = 1
            y_span = 1

            # --- 우측 병합 검사 ---
            # 먼저 시작 셀의 오른쪽 경계선을 검사
            if not is_vertical_boundary_present(table_img, x_list[j], y_start,
                                                x_list[j+1]-x_list[j],
                                                y_list[i+1]-y_start):
                # 시작 셀에 경계선이 없으면 병합 시작
                k = j + 1
                # while문: 현재 셀부터 우측으로 진행
                while k < n_cols:
                    # 검사 대상: 셀 (i, k)의 오른쪽 경계선
                    candidate_width = x_list[k+1] - x_list[k]
                    if is_vertical_boundary_present(table_img, x_list[k],
                                                    y_start,
                                                    candidate_width,
                                                    y_list[i+1]-y_start):
                        # 경계선 발견 → 포함시키고 종료
                        k += 1
                        break
                    else:
                        k += 1
                x_span = k - j  # 병합한 셀 수 (최소 1)
            else:
                x_span = 1

            # --- 아래쪽 병합 검사 ---
            # 먼저 시작 셀의 아래쪽 경계선을 검사
            if not is_horizontal_boundary_present(table_img, x_list[j],
                                                  y_list[i],
                                                  x_list[j+1]-x_list[j],
                                                  y_list[i+1]-y_list[i]):
                r = i + 1
                while r < n_rows:
                    can_merge = True
                    for col in range(j, j + x_span):
                        candidate_width = x_list[col+1] - x_list[col]
                        if is_horizontal_boundary_present(table_img, x_list[col],
                                                          y_list[r],
                                                          candidate_width,
                                                          y_list[r+1]-y_list[r]):
                            # 경계선 발견 → 해당 행 포함 후 종료
                            r += 1
                            can_merge = False
                            break
                    if not can_merge:
                        break
                    else:
                        r += 1
                y_span = r - i
            else:
                y_span = 1

            # 최종 영역 좌표 계산
            x_end = x_list[j + x_span]
            y_end = y_list[i + y_span]

            # Mark 병합된 영역의 모든 셀을 visited 처리
            for r_idx in range(i, i + y_span):
                for c_idx in range(j, j + x_span):
                    visited[r_idx][c_idx] = True

            # 활성 셀 객체 생성 (좌측 상단 셀에만 저장)
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

# 예시 사용
if __name__ == '__main__':
    image_path = 'table_region_1.png'
    # 예시: x_list와 y_list (마지막 값은 표의 우측, 하단 경계)
    x_list = [0, 50, 100, 150, 200]      # 기본 4열
    y_list = [0, 30, 60, 90, 120]          # 기본 4행

    cell_matrix, cells = generate_table_structure(image_path, x_list, y_list)
    
    # 디버그: 활성 셀(병합 셀의 좌측 상단 셀) 정보 출력
    for cell in cells:
        print(cell)
