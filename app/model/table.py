# app/model/table.py
import cv2
import numpy as np

class Table():
    def __init__(self, table_image_path, x_coords, y_coords, page_num, table_idx):
        """테이블 이미지 로드 및 기본 정보 초기화"""
        table_img = cv2.imread(table_image_path)
        if table_img is None:
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {table_image_path}")

        self.table_image = table_img
        self.x_coords, self.y_coords = x_coords, y_coords
        self.n_cols, self.n_rows = len(x_coords) - 1, len(y_coords) - 1
        self.cells = []
        self.cell_matrix = [[None] * self.n_cols for _ in range(self.n_rows)]
        self.page_num, self.table_idx = page_num, table_idx
    
    class Cell():
        def __init__(self, row, col, x_start, y_start, x_end, y_end, data, 
                    is_merged=False, group=[]):
            """셀의 위치, 데이터, 병합 여부 초기화"""
            self.row, self.col = row, col
            self.x_start, self.y_start, self.x_end, self.y_end = x_start, y_start, x_end, y_end
            self.data = data
            self.is_merged = is_merged
            if is_merged:
                self.area = self.set_area(group)
                self.merge_cols = self.set_merge_cols(group)
                self.unmerged_coords = self.set_unmerged_coords(group, self.area)

        def set_area(self, group):
            """병합된 셀의 영역 계산
            예:
                group = [(1,1), (1,2), (2,1), (2,2)]
                반환: ((1, 2), (1, 2))  # (행 범위, 열 범위)
            """
            r_list, c_list = map(list, zip(*group))
            return ((np.min(r_list), np.max(r_list)), (np.min(c_list), np.max(c_list)))

        def set_merge_cols(self, group):
            """병합된 열을 연속된 그룹으로 정리
            예:
                group = [(2,1), (2,2), (2,3), (2,5), (2,6), (3,2), (3,3)]
                반환: [(2, (1, 3)), (2, (5, 6)), (3, (2, 3))]
            """
            pairs_sorted = sorted(group, key=lambda x: (x[0], x[1]))
            groups, current_group = [], []

            for pair in pairs_sorted:
                if not current_group or (pair[0] == current_group[-1][0] and pair[1] == current_group[-1][1] + 1):
                    current_group.append(pair)
                else:
                    groups.append((current_group[0][0], (current_group[0][1], current_group[-1][1])))
                    current_group = [pair]

            if current_group:
                groups.append((current_group[0][0], (current_group[0][1], current_group[-1][1])))

            return groups

        def set_unmerged_coords(self, group, area):
            """병합된 영역 내에서 개별 셀 좌표 반환
            예:
                group = [(1,1), (1,2)]
                area = ((1, 2), (1, 2))
                반환: [(2,1), (2,2)]
            """
            (min_r, max_r), (min_c, max_c) = area
            return [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1) if (r, c) not in set(group)]
