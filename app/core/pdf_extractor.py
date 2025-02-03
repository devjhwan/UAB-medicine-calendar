# pdf_extractor.py
import os
from pdf2image import convert_from_path

def parse_pdf_to_images(pdf_path, image_dir='images', dpi_value=300):
    """
    PDF 파일의 각 페이지를 지정된 DPI로 이미지로 변환하여,
    'images' 디렉토리에 "page_1.png", "page_2.png", … 형식으로 저장한다.
    저장된 이미지 파일 경로들을 리스트 형태로 반환한다.
    """
    os.makedirs(image_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi_value)
    output_paths = []
    for i, image in enumerate(images):
        output_path = os.path.join(image_dir, f'page_{i+1}.png')
        image.save(output_path, 'PNG')
        print(f"Saved PDF page image: {output_path}")
        output_paths.append(output_path)
    return output_paths

# 예시 사용
if __name__ == '__main__':
    pdf_path = 'your_document.pdf'
    paths = parse_pdf_to_images(pdf_path, image_dir='images', dpi_value=300)
    print("저장된 이미지 파일 경로:")
    print(paths)
