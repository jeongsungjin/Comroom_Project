import pyrealsense2 as rs
import cv2
import numpy as np
import os

# 저장 폴더
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

# RealSense pipeline 생성
pipeline = rs.pipeline()
config = rs.config()

# 1280x720 컬러 스트림 설정 (YOLO 데이터셋용 추천)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 카메라 시작
pipeline.start(config)

print("=== RealSense D435i RGB 모드 시작됨 ===")
print("SPACE: 이미지 저장, ESC: 종료")

count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("D435i Color", color_image)

        key = cv2.waitKey(1)

        # ESC → 종료
        if key == 27:
            break

        # Space → 이미지 저장
        if key == 32:
            filepath = os.path.join(save_dir, f"img_{count}.jpg")
            cv2.imwrite(filepath, color_image)
            print("Saved:", filepath)
            count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
