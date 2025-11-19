import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.absolute()

# 모델 가중치 경로
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# 실시간 추론 해상도를 유지하면서도 8GB급 GPU에서 학습이 가능하도록
# 기본 모델을 yolov8m으로 변경 (필요 시 YOLO_MODEL 환경변수로 재정의)
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8m.pt")
PRETRAINED_WEIGHTS = WEIGHTS_DIR / MODEL_NAME

# 프리트레인 모델 가중치 (없으면 자동 다운로드)
if not PRETRAINED_WEIGHTS.exists():
    print(f"프리트레인 모델을 다운로드합니다: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)  # 자동 다운로드
else:
    model = YOLO(str(PRETRAINED_WEIGHTS))

# 데이터셋 설정 파일
DATA_YAML = PROJECT_ROOT / "data.yaml"

if not DATA_YAML.exists():
    print(f"경고: {DATA_YAML} 파일이 없습니다. data.yaml 템플릿을 참고하여 생성하세요.")
    exit(1)

# 학습 시작
print("\n" + "="*60)
print("🚀 YOLO v8 파인튜닝 시작")
print("="*60)
print(f"📦 프리트레인 모델: {MODEL_NAME}")
print(f"📁 데이터셋 설정: {DATA_YAML}")
print(f"📊 학습 이미지: train/images")
print(f"📊 검증 이미지: valid/images")
print("="*60 + "\n")

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=1280,  # 실시간 추론 해상도(1280x720)를 맞추기 위해 유지
    batch=1,  # GPU 메모리 확보를 위해 배치 1
    device='0',   # GPU 사용 시 '0' 또는 0, CPU면 'cpu' (CUDA 확인 후 수정 가능)
    workers=2,  # dataloader 메모리 사용 최소화
    rect=True,  # 16:9 비율을 최대한 유지해 패딩 낭비 감소
    plots=True,  # 학습 결과 그래프 생성
    save=True,
    save_period=10,  # 10 에포크마다 체크포인트 저장
    amp=True,  # CPU에서는 AMP 비활성화
)

# 학습 완료 후 최고 모델을 weights 디렉토리로 복사
if results and hasattr(results, 'save_dir'):
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        dest_model = WEIGHTS_DIR / "best.pt"
        shutil.copy2(best_model, dest_model)
        print(f"\n✅ 학습된 모델이 복사되었습니다: {dest_model}")
        print(f"   원본 위치: {best_model}")
    else:
        print(f"\n⚠️  최고 모델을 찾을 수 없습니다: {best_model}")
else:
    # results가 없거나 save_dir이 없는 경우 기본 경로 확인
    default_best = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if default_best.exists():
        dest_model = WEIGHTS_DIR / "best.pt"
        shutil.copy2(default_best, dest_model)
        print(f"\n✅ 학습된 모델이 복사되었습니다: {dest_model}")
    else:
        print(f"\n⚠️  학습된 모델을 찾을 수 없습니다. 수동으로 확인하세요:")
        print(f"   예상 위치: {default_best}")