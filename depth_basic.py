import pyrealsense2 as rs
import numpy as np
import cv2

# 1) 파이프라인 생성
pipeline = rs.pipeline()
config = rs.config()

# 2) 스트림 설정 (기본)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 3) 시작
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        # Depth 시각화
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow("Color", color)
        cv2.imshow("Depth", depth_colormap)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
