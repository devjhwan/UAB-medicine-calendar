import cv2

class Table():
    def __init__(self, table_image_path, x_coords, y_coords, page_num, table_idx):
        table_img = cv2.imread(table_image_path)
        if table_img is None:
            raise ValueError("이미지 파일을 찾을 수 없습니다: " + table_image_path)
        self.table_image = table_img
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.n_cols = len(x_coords) - 1
        self.n_rows = len(y_coords) - 1
        self.cell_matrix = [[None for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        self.page_num = page_num
        self.table_idx = table_idx
    
    class Cell():
        def __init__(self, row, col, x_start, y_start, x_end, y_end, data, \
                        is_merged=False, area=None, group_idx=0):
            self.row = row
            self.col = col
            self.x_start = x_start
            self.y_start = y_start
            self.x_end = x_end
            self.y_end = y_end
            self.data = data
            self.is_merged = is_merged
            self.area = area
            self.group_idx = group_idx