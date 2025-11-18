#!/usr/bin/env python3
"""
데이터셋 검증 스크립트
학습 전에 데이터셋이 올바르게 구성되었는지 확인
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.absolute()

print("="*60)
print("📋 데이터셋 검증")
print("="*60)

# data.yaml 확인
data_yaml = PROJECT_ROOT / "data.yaml"
if not data_yaml.exists():
    print(f"❌ data.yaml 파일을 찾을 수 없습니다: {data_yaml}")
    exit(1)

print(f"✅ data.yaml 파일 확인: {data_yaml}")

# data.yaml 로드
with open(data_yaml, 'r') as f:
    data = yaml.safe_load(f)

print(f"\n📊 데이터셋 정보:")
print(f"   클래스 개수: {data.get('nc', 'N/A')}")
print(f"   클래스 목록: {data.get('names', [])}")

# 경로 확인
path = Path(data.get('path', PROJECT_ROOT))
train_path = path / data.get('train', '')
val_path = path / data.get('val', '')

print(f"\n📁 경로 확인:")
print(f"   데이터셋 루트: {path}")
print(f"   학습 경로: {train_path}")
print(f"   검증 경로: {val_path}")

# 이미지 파일 확인
train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))

print(f"\n🖼️  이미지 파일:")
print(f"   학습 이미지: {len(train_images)}개")
print(f"   검증 이미지: {len(val_images)}개")

# 라벨 파일 확인
train_labels_dir = train_path.parent / "labels"
val_labels_dir = val_path.parent / "labels"

train_labels = list(train_labels_dir.glob("*.txt")) if train_labels_dir.exists() else []
val_labels = list(val_labels_dir.glob("*.txt")) if val_labels_dir.exists() else []

print(f"\n📝 라벨 파일:")
print(f"   학습 라벨: {len(train_labels)}개")
print(f"   검증 라벨: {len(val_labels)}개")

# 이미지-라벨 매칭 확인
if len(train_images) != len(train_labels):
    print(f"\n⚠️  경고: 학습 이미지({len(train_images)})와 라벨({len(train_labels)}) 개수가 일치하지 않습니다!")
else:
    print(f"\n✅ 학습 이미지-라벨 매칭 확인")

if len(val_images) != len(val_labels):
    print(f"⚠️  경고: 검증 이미지({len(val_images)})와 라벨({len(val_labels)}) 개수가 일치하지 않습니다!")
else:
    print(f"✅ 검증 이미지-라벨 매칭 확인")

# 샘플 라벨 확인
if train_labels:
    sample_label = train_labels[0]
    print(f"\n📄 샘플 라벨 확인: {sample_label.name}")
    with open(sample_label, 'r') as f:
        lines = f.readlines()
        print(f"   라벨 개수: {len(lines)}개")
        if lines:
            print(f"   첫 번째 라벨: {lines[0].strip()}")

print("\n" + "="*60)
if len(train_images) > 0 and len(val_images) > 0:
    print("✅ 데이터셋 검증 완료! 학습을 시작할 수 있습니다.")
    print("   실행: python3 train.py")
else:
    print("❌ 데이터셋에 문제가 있습니다. 위의 경고를 확인하세요.")
print("="*60)

