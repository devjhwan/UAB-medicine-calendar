# UAB Medicine Calendar

## 소개

UAB Medicine Calendar는 **Universitat Autònoma de Barcelona (UAB) 의과대학 일정표(PDF 형식)**에서  
표 데이터를 자동으로 추출하고 분석하는 도구입니다.  
이 프로젝트는 **UAB 의과대학의 공식 일정표**를 효율적으로 관리하고 활용하기 위해 개발되었습니다.  
PDF 문서에서 표를 검출하고, 각 셀의 데이터를 추출하여 일정 관리를 자동화할 수 있도록 지원합니다.

이 도구는 다음과 같은 기능을 제공합니다:

- **PDF를 고해상도 이미지로 변환**  
  - PDF 일정표의 각 페이지를 이미지로 변환하여 처리합니다.
  
- **표 영역 검출 및 셀 좌표 추출**  
  - Adaptive thresholding 및 컨투어 검출을 활용하여 일정표의 표 구조를 분석합니다.
  
- **셀 병합 여부 감지 및 데이터 정렬**  
  - 병합된 셀을 감지하여 정확한 일정 데이터를 제공합니다.

이 프로젝트는 **Universitat Autònoma de Barcelona (UAB) 의과대학 일정표에서만 사용 가능**하며,  
다른 대학교 또는 기관의 일정표에서는 정확한 결과를 보장하지 않습니다.

## 설치

### 1. 저장소 클론

다음 명령어를 실행하여 저장소를 클론합니다:

```
git clone https://github.com/devjhwan/UAB-medicine-calendar.git
cd UAB-medicine-calendar
```
### 2. 가상 환경 설정 (선택 사항)

Python 가상 환경을 사용하는 경우, 다음 명령어를 실행하여 환경을 설정합니다:

```
python -m venv venv
source venv/bin/activate  # (Windows의 경우: venv\Scripts\activate)
```

### 3. 필수 패키지 설치

프로젝트 실행에 필요한 패키지를 설치합니다:

```
pip install -r requirements.txt
```

### 4. 추가 의존성 설치

이 프로젝트는 pdf2image 및 pytesseract 라이브러리를 사용합니다.
이 패키지를 정상적으로 실행하려면 추가적인 의존성 설치가 필요합니다.

❗ Windows 사용자:
1. Poppler 설치

Poppler 공식 다운로드 페이지에서 poppler-xx_x_x.zip 파일을 다운로드합니다.
압축을 풀고, bin 폴더의 경로를 환경 변수(PATH)에 추가합니다.
예를 들어, C:\poppler-xx_x_x\bin 폴더 경로를 추가해야 합니다.

2. Tesseract 설치

Tesseract 공식 다운로드 페이지에서 Windows용 설치 파일을 다운로드하여 설치합니다.
설치 후 Tesseract-OCR 폴더 경로를 환경 변수(PATH)에 추가합니다.
기본 경로는 C:\Program Files\Tesseract-OCR입니다.

❗ Linux 사용자 (Ubuntu 기준):
1. Poppler 설치
다음 명령어를 실행하여 Poppler를 설치합니다:

```
sudo apt update
sudo apt install poppler-utils
```

2. Tesseract 설치
다음 명령어를 실행하여 Tesseract를 설치합니다:

```
sudo apt install tesseract-ocr
```

### 5. 실행
설치가 완료되면 다음 명령어로 애플리케이션을 실행할 수 있습니다:

```
python app/main.py
```

## 기술 스택

이 프로젝트는 **Python 기반의 이미지 처리 및 OCR(Optical Character Recognition) 기술**을 활용하여  
PDF 일정표를 분석하고 데이터를 추출합니다.  
주요 기술 스택은 다음과 같습니다:

### 📌 언어 & 환경
- **Python** 3.x

### 📌 주요 라이브러리
- **pdf2image**: PDF 문서를 고해상도 이미지로 변환하는 라이브러리
- **pytesseract**: Google Tesseract OCR을 이용하여 이미지에서 텍스트를 추출하는 라이브러리
- **OpenCV**: 이미지 전처리 및 표 검출을 위한 컴퓨터 비전 라이브러리
- **NumPy**: 행렬 연산 및 이미지 데이터 처리
- **Pandas**: 추출된 데이터를 표 형식으로 정리하고 다룰 때 사용
- **Matplotlib** (선택 사항): 이미지 디버깅 및 시각화 도구

### 📌 OCR & 이미지 처리
- **Tesseract-OCR**: Google의 오픈소스 OCR 엔진으로 텍스트 추출 수행
- **Poppler**: PDF를 이미지로 변환하는 데 사용되는 오픈소스 라이브러리
- **Adaptive Thresholding**: 표 검출을 위한 이진화 기법
- **Contour Detection**: 표의 경계를 감지하여 셀을 추출하는 기술

### 📌 실행 환경
- Windows 및 Linux(Ubuntu) 지원
- 가상 환경(Virtual Environment) 사용 가능

## 🚧 개발 중인 사항 (TODO)

현재 프로젝트는 개선 및 최적화 작업이 진행 중이며, 다음과 같은 기능이 추가될 예정입니다:

### ✅ 1. OCR 성능 개선
- Tesseract OCR의 설정 최적화 (예: `--psm` 옵션 조정)
- 일정표의 특정 폰트와 레이아웃에 대한 OCR 정확도 향상
- 사전(preprocessing) 및 후처리(postprocessing) 알고리즘 추가

### ✅ 2. 표 구조 인식 강화
- 복잡한 병합 셀(Merged Cell) 처리 로직 개선
- 표 경계 검출(Contour Detection) 알고리즘 성능 최적화
- 행(row)과 열(column) 자동 정렬 기능 추가

### ✅ 3. 에러 핸들링 및 예외 처리
- OCR 결과가 예상과 다를 경우의 예외 처리 추가
- 지원하지 않는 PDF 형식에 대한 안내 메시지 제공
- 오류 로그(logging) 기능 추가

### ✅ 4. PDF 파일 처리 속도 향상
- PDF → 이미지 변환 시 최적 DPI 설정 적용
- OpenCV 및 NumPy 연산 최적화를 통한 속도 개선
- 멀티스레딩 또는 병렬 처리 적용 가능성 검토

### ✅ 5. 사용자 편의 기능 추가
- GUI 또는 CLI(Command Line Interface) 옵션 제공
- 일정 데이터를 CSV 또는 Excel로 내보내는 기능 추가
- 실행 결과를 시각적으로 확인할 수 있는 디버깅 툴 제공

이 프로젝트는 계속해서 발전하고 있으며, 위의 항목들은 변경될 수 있습니다.


## 연락처
질문이나 문의 사항이 있으시면 devjhwan@example.com으로 연락주세요.
