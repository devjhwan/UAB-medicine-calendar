# table.py (예: app/core/table.py 또는 원하는 위치에 생성)
import cv2
import numpy as np
from core.table_structure import generate_table_structure

class Table:
    """
    Table 클래스는 표의 셀 그리드(cell_matrix)와 병합 상태(merge_matrix)를 저장한다.
    
    Attributes:
      - table_image_path (str): 원본 표 이미지 파일 경로.
      - x_coords (list): x 좌표 리스트 (길이: n+1)
      - y_coords (list): y 좌표 리스트 (길이: m+1)
      - cell_matrix (list of list): n x m 크기의 2차원 리스트.
            각 원소는 독립 셀의 경우 딕셔너리로, 병합 셀 그룹의 하위 셀은 None.
            딕셔너리에는 최소한 'row', 'col', 'data' 필드가 포함된다.
      - merge_matrix (list of list): cell_matrix와 동일한 크기의 2차원 리스트.
            각 원소는 해당 위치 셀이 어느 병합 그룹에 속하는지를 나타내며,
            독립 셀은 0, 병합 그룹은 1부터 시작하는 그룹 인덱스로 표시된다.
    """
    def __init__(self, image_path, x_coords, y_coords, cell_matrix, cells, merge_matrix):
        self.table_image_path = image_path
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.cell_matrix = cell_matrix
        self.cells = cells
        self.merge_matrix = merge_matrix

    @classmethod
    def from_image(cls, image_path, x_coords, y_coords, intensity_threshold=50):
        """
        주어진 이미지 파일 경로와 x, y 좌표 리스트를 기반으로 셀 그리드(cell_matrix)와 병합 상태(merge_matrix)를 생성한다.
        
        Parameters:
          - image_path (str): 원본 표 이미지 파일 경로.
          - x_coords (list): x 좌표 리스트 (예: [x0, x1, ..., xn])
          - y_coords (list): y 좌표 리스트 (예: [y0, y1, ..., ym])
          - intensity_threshold (int): 경계 판별 임계값 (기본값 50)
          
        내부적으로 generate_table_structure 함수를 호출하여, 셀 그리드와 병합 그룹 정보를 생성한 후
        Table 인스턴스를 반환한다.
        """
        cell_matrix, cells, merge_matrix = generate_table_structure(image_path, x_coords, y_coords, intensity_threshold)
        return cls(image_path, x_coords, y_coords, cell_matrix, cells, merge_matrix)
    
    def __str__(self):
        """
        Table 객체의 요약 정보를 문자열로 반환한다.
        """
        n = len(self.cell_matrix)
        m = len(self.cell_matrix[0]) if n > 0 else 0
        return f"Table: {n}x{m} cells; x_coords: {self.x_coords}; y_coords: {self.y_coords}"

# 예시 사용 (테스트용)
if __name__ == '__main__':
    # 예시: 좌표 리스트 (실제 값은 표 추출 단계 결과에 따라 결정)
    x_coords = [0, 50, 100, 150, 200]
    y_coords = [0, 30, 60, 90, 120]
    image_path = 'tables/page_1/table_region_20.png'
    
    table = Table.from_image(image_path, x_coords, y_coords)
    print(table)
    
    # cell_matrix 출력 (디버깅 용)
    print("Cell Matrix:")
    for row in table.cell_matrix:
        print(row)
    
    # merge_matrix 출력 (디버깅 용)
    print("Merge Matrix:")
    for row in table.merge_matrix:
        print(row)
