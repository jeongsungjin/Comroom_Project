#!/usr/bin/env python3
"""
YOLO v8 프리트레인 모델 추론 테스트 스크립트
카메라가 없어도 테스트 이미지로 확인 가능
"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.absolute()
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolov8l.pt"

print("=" * 60)
print("YOLO v8 프리트레인 모델 추론 테스트")
print("=" * 60)

# 모델 로드
if not WEIGHTS_PATH.exists():
    print(f"❌ 모델 파일을 찾을 수 없습니다: {WEIGHTS_PATH}")
    print("   먼저 모델을 다운로드하세요:")
    print("   python3 -c \"from ultralytics import YOLO; YOLO('yolov8l.pt')\"")
    sys.exit(1)

print(f"📦 모델 로드 중: {WEIGHTS_PATH}")
model = YOLO(str(WEIGHTS_PATH))
print("✅ 모델 로드 완료")

# 카메라 테스트
print("\n📹 카메라 연결 테스트...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✅ 카메라 연결 성공!")
        print(f"   해상도: {frame.shape[1]}x{frame.shape[0]}")
        
        # 추론 테스트
        print("\n🔍 추론 테스트 중...")
        results = model.predict(frame, imgsz=640, conf=0.5, device='cpu', verbose=False)
        
        # 결과 확인
        annotated = results[0].plot()
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"✅ 탐지된 객체 수: {num_detections}")
        
        if num_detections > 0:
            print("\n📊 탐지 결과:")
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                print(f"  {i+1}. {name} (신뢰도: {conf:.2f}) - 중심: ({cx:.1f}, {cy:.1f})")
        
        # 결과 이미지 저장
        output_path = PROJECT_ROOT / "test_inference_result.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 결과 이미지 저장: {output_path}")
        
        print("\n✅ 실시간 추론 스크립트가 정상 작동합니다!")
        print("   다음 명령어로 실시간 추론을 실행하세요:")
        print("   python3 realtime_infer.py --weights weights/yolov8l.pt --show --fps")
    else:
        print("⚠️  카메라에서 프레임을 읽을 수 없습니다")
    cap.release()
else:
    print("⚠️  카메라를 열 수 없습니다 (카메라가 연결되어 있지 않거나 다른 프로그램이 사용 중)")
    print("\n📸 테스트 이미지로 추론 테스트...")
    
    # dataset 폴더의 이미지로 테스트
    test_images = list((PROJECT_ROOT / "dataset").glob("*.jpg"))
    if test_images:
        test_img = test_images[0]
        print(f"   테스트 이미지: {test_img.name}")
        img = cv2.imread(str(test_img))
        
        results = model.predict(img, imgsz=640, conf=0.5, device='cpu', verbose=False)
        annotated = results[0].plot()
        
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"✅ 탐지된 객체 수: {num_detections}")
        
        output_path = PROJECT_ROOT / "test_inference_result.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"💾 결과 이미지 저장: {output_path}")
        print("\n✅ 추론 스크립트가 정상 작동합니다!")
    else:
        print("❌ 테스트할 이미지를 찾을 수 없습니다")

print("\n" + "=" * 60)

