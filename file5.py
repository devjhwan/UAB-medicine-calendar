import cv2
import numpy as np

def generate_debug_image(image_path, cell_matrix, output_path='debug_table.png'):
    """
    주어진 이미지 경로와 셀 행렬(cell_matrix)을 이용해 디버그용 이미지를 생성한다.
    각 활성 셀(병합 셀의 좌측 상단 셀)에 대해 사각형과 x_span, y_span 정보를 표시한다.
    
    Parameters:
        image_path (str): 원본 테이블 이미지 경로.
        cell_matrix (list of list): generate_table_structure 함수 등에서 생성한 셀 행렬.
        output_path (str): 디버그 이미지가 저장될 경로 (기본값 'debug_table.png').
    """
    # 원본 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
    
    # cell_matrix의 각 활성 셀에 대해 사각형과 텍스트 표시
    for i, row in enumerate(cell_matrix):
        for j, cell in enumerate(row):
            if cell is not None:
                x = cell['x_start']
                y = cell['y_start']
                w = cell['x_length']
                h = cell['y_length']
                x_span = cell['x_span']
                y_span = cell['y_span']
                
                # 사각형 그리기 (빨간색, 두께 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # x_span, y_span 정보를 파란색 텍스트로 표시
                text = f"{x_span}x{y_span}"
                cv2.putText(image, text, (x + 5, y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 디버그 이미지 저장
    cv2.imwrite(output_path, image)
    print("디버그 이미지가 저장되었습니다:", output_path)


# 예시 사용
if __name__ == '__main__':
    # 이전에 생성한 셀 행렬(cell_matrix)을 사용한다고 가정
    # 예시: image_path, x_list, y_list를 이용해 셀 행렬 생성
    image_path = 'tables/page_1/table_region_20.png'
    # 실제 값은 표 추출 시 결정됨 (예시)
    x_list = [3, 152, 296, 439, 583, 726, 870, 1019, 1169, 1318, 1468, 1617, 1761, 1904, 2048, 2192, 2335]
    y_list = [3, 54, 105, 156, 207, 258, 309, 360, 411, 462, 513, 564, 615, 666, 717, 768, 819, 870, 921, 972, 1023, 1074, 1125, 1178, 1229, 1280, 1331, 1383, 1433, 1484, 1535, 1586, 1637, 1691, 1744, 1795, 1847, 1897, 1948, 1999, 2050, 2101, 2152, 2203, 2255, 2305, 2356, 2407, 2458, 2509, 2560, 2611, 2662]
    
    # generate_table_structure 함수는 앞서 구현한 셀 행렬 생성 함수
    # (cell_matrix, cells) = generate_table_structure(image_path, x_list, y_list)
    # 여기서는 예시로 간단한 cell_matrix를 가정하고 진행
    # 실제 코드에서는 앞서 구현한 generate_table_structure 함수를 호출하세요.
    
    # 예시: generate_table_structure 함수를 호출했다고 가정하고 cell_matrix를 받아옴.
    from file4 import generate_table_structure  # 가정: 별도 모듈로 구현됨
    cell_matrix, cells = generate_table_structure(image_path, x_list, y_list)
    
    # 디버그 이미지 생성
    generate_debug_image(image_path, cell_matrix, output_path='debug_table.png')
