from pdf2image import convert_from_path
import os

def parse_pdf_to_images(pdf_path, dpi_value=300):
    """
    PDF 파일을 이미지로 변환하여 "images" 디렉토리에 저장합니다.
    
    Parameters:
        pdf_path (str): 변환할 PDF 파일의 경로.
        dpi_value (int): 이미지 해상도 (DPI, 기본값은 300).
        
    변환된 이미지는 "images" 디렉토리에 "page_1.png", "page_2.png", ... 형식으로 저장됩니다.
    """
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=dpi_value)
    for i, image in enumerate(images):
        output_path = os.path.join(image_dir, f'page_{i+1}.png')
        image.save(output_path, 'PNG')
        print(f"Saved: {output_path}")

# 예시 사용법:
# parse_pdf_to_images('DOC_HorariMed3_VH,0.pdf', dpi_value=300)
