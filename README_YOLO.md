# YOLO v8 학습 및 실시간 추론 환경 설정 가이드

## 📋 목차
1. [환경 설정](#환경-설정)
2. [프리트레인 모델 다운로드](#프리트레인-모델-다운로드)
3. [데이터셋 준비](#데이터셋-준비)
4. [학습 실행](#학습-실행)
5. [실시간 추론 실행](#실시간-추론-실행)

---

## 🔧 환경 설정

### 1. 가상환경 활성화 (이미 있는 경우)
```bash
cd /home/kim/comroom
source venv/bin/activate
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. GPU 사용 시 (선택사항)
CUDA가 설치되어 있다면 PyTorch가 자동으로 GPU를 사용합니다.
```bash
# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📥 프리트레인 모델 다운로드

### 방법 1: 자동 다운로드 (권장)
`train.py`를 실행하면 자동으로 `yolov8l.pt`가 다운로드됩니다.

### 방법 2: 수동 다운로드
```bash
cd /home/kim/comroom/weights
# YOLO v8 Large 모델 다운로드
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
```

또는 Python으로:
```python
from ultralytics import YOLO
model = YOLO("yolov8l.pt")  # 자동 다운로드
```

---

## 📁 데이터셋 준비

### 1. 데이터셋 구조
YOLO 형식의 데이터셋 구조:
```
dataset/
├── images/
│   ├── train/
│   │   ├── img_0.jpg
│   │   ├── img_1.jpg
│   │   └── ...
│   └── val/
│       ├── img_10.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_0.txt
    │   ├── img_1.txt
    │   └── ...
    └── val/
        ├── img_10.txt
        └── ...
```

### 2. 라벨 파일 형식
각 이미지에 대응하는 `.txt` 파일이 필요합니다.
- 형식: `class_id center_x center_y width height`
- 좌표는 정규화된 값 (0.0 ~ 1.0)
- 예시: `0 0.5 0.5 0.2 0.3` (클래스 0, 중심 (0.5, 0.5), 크기 0.2x0.3)

### 3. data.yaml 수정
`data.yaml` 파일을 열어서 실제 데이터셋 경로와 클래스 정보를 수정하세요:
```yaml
path: /home/kim/comroom
train: dataset/images/train
val: dataset/images/val
nc: 14  # 클래스 개수
names: [...]  # 클래스 이름 목록
```

---

## 🚀 학습 실행

### 기본 실행
```bash
python train.py
```

### 학습 파라미터 수정
`train.py` 파일을 열어서 다음 파라미터를 수정할 수 있습니다:
- `epochs`: 학습 에포크 수 (기본: 50)
- `imgsz`: 입력 이미지 크기 (기본: 640)
- `batch`: 배치 크기 (기본: 4, GPU 메모리에 따라 조정)
- `device`: GPU 사용 시 `0`, CPU 사용 시 `'cpu'`
- `workers`: 데이터 로더 워커 수 (기본: 2)

### 학습 결과
학습이 완료되면 다음 위치에 모델이 저장됩니다:
- `runs/detect/train/weights/best.pt` (최고 성능 모델)
- `runs/detect/train/weights/last.pt` (마지막 체크포인트)

---

## 🎥 실시간 추론 실행

### 기본 실행 (웹캠 사용)
```bash
python realtime_infer.py --show --fps
```

### 학습된 모델 사용
```bash
python realtime_infer.py \
    --weights runs/detect/train/weights/best.pt \
    --show \
    --fps \
    --log-every 30
```

### 주요 옵션
- `--weights`: 모델 가중치 경로 (기본: `weights/best.pt` 또는 `weights/yolov8l.pt`)
- `--source`: 비디오 소스 (기본: `0` = 웹캠, 파일 경로나 URL도 가능)
- `--show`: 화면에 결과 표시
- `--fps`: FPS 표시
- `--conf`: 신뢰도 임계값 (기본: 0.5)
- `--iou`: NMS IoU 임계값 (기본: 0.7)
- `--log-every N`: N 프레임마다 탐지 결과 로그 출력
- `--save PATH`: 결과를 비디오 파일로 저장
- `--homography PATH`: 호모그래피 변환 파일 경로 (BEV 좌표 변환용)

### 예시
```bash
# 웹캠으로 실시간 추론 (화면 표시 + FPS)
python realtime_infer.py --show --fps

# 학습된 모델로 추론 + 결과 저장
python realtime_infer.py \
    --weights runs/detect/train/weights/best.pt \
    --show \
    --save output.mp4

# 특정 비디오 파일로 추론
python realtime_infer.py \
    --source video.mp4 \
    --weights weights/best.pt \
    --show
```

---

## 🔍 문제 해결

### 1. 모델 파일을 찾을 수 없음
- `weights/` 디렉토리에 모델 파일이 있는지 확인
- `train.py` 실행 시 자동 다운로드됨

### 2. 데이터셋을 찾을 수 없음
- `data.yaml`의 경로가 올바른지 확인
- 상대 경로는 `path` 기준으로 해석됨

### 3. GPU가 인식되지 않음
```bash
# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. 카메라가 열리지 않음
- 카메라 인덱스 확인: `--source 0`, `--source 1` 등 시도
- 카메라 권한 확인 (Linux)
- 다른 프로그램이 카메라를 사용 중인지 확인

---

## 📝 참고사항

- 학습된 모델은 `runs/detect/train/weights/` 디렉토리에 저장됩니다
- 실시간 추론 시 클래스별 위치 좌표가 콘솔에 출력됩니다
- 호모그래피 파일을 제공하면 BEV(Bird's Eye View) 좌표도 변환됩니다

