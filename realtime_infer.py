import argparse
import os
import sys
import time
from typing import Union, Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("경고: pyrealsense2가 설치되지 않았습니다. RealSense 카메라를 사용할 수 없습니다.")


WINDOW_TITLE = "YOLOv8 Realtime Inference"


def parse_args() -> argparse.Namespace:
    from pathlib import Path
    
    # 기본 가중치 경로 (프로젝트 루트 기준)
    project_root = Path(__file__).parent.absolute()
    default_weights = project_root / "weights" / "best.pt"  # 학습된 모델
    if not default_weights.exists():
        default_weights = project_root / "weights" / "yolov8l.pt"  # 프리트레인 모델
    
    parser = argparse.ArgumentParser(description="YOLOv8 실시간 추론")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="realsense",  # "realsense" 또는 "rs" = RealSense 카메라, 숫자 = 웹캠 인덱스
        help="비디오 소스: 'realsense' 또는 'rs' = RealSense D435i, 숫자 = 웹캠 인덱스, 경로 = 비디오 파일"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="추론 입력 해상도 (1280x720 해상도 사용)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",  # "auto" 선택 시 가용 GPU 자동 사용
        help="추론 장치 ('auto', 'cpu', 'cuda:0', '0', 'mps' 등)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    parser.add_argument(
        "--fps",
        action="store_true",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--homography",
        type=str,
        default="",
    )
    return parser.parse_args()


def to_int_if_digit(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def resolve_device_argument(device_arg: str) -> str:
    requested = (device_arg or "").strip()
    normalized = requested.lower()
    if normalized and normalized not in {"auto", "autodetect"}:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    mps_available = (
        hasattr(torch.backends, "mps")
        and hasattr(torch.backends.mps, "is_available")
        and torch.backends.mps.is_available()
    )
    if mps_available:
        return "mps"

    return "cpu"


def is_gui_available() -> bool:
    if os.name == "nt":
        return True
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def load_homography(path: str) -> Union[np.ndarray, None]:
    if not path:
        return None
    try:
        if path.lower().endswith(".npy"):
            H = np.load(path)
        else:
            import yaml

            with open(path, "r") as f:
                data = yaml.safe_load(f)
            H = np.asarray(data.get("H"), dtype=np.float64)
        if H.shape == (3, 3):
            return H
    except Exception:
        pass
    print("호모그래피 파일을 불러오지 못했습니다. 경로/형식을 확인하세요.")
    return None


def apply_homography_point(H: np.ndarray, u: float, v: float) -> Union[tuple, None]:
    if H is None:
        return None
    pt = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
    p = H @ pt
    if p[2, 0] == 0:
        return None
    Xw = (p[0, 0] / p[2, 0])
    Yw = (p[1, 0] / p[2, 0])
    return (float(Xw), float(Yw))


def init_realsense() -> Optional[rs.pipeline]:
    """RealSense 카메라 초기화"""
    if not REALSENSE_AVAILABLE:
        print("❌ pyrealsense2가 설치되지 않았습니다.")
        print("   설치: pip install pyrealsense2")
        return None
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # RealSense D435i 스트림 설정
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        pipeline.start(config)
        print("✅ RealSense D435i 카메라 연결 성공")
        return pipeline
    except Exception as e:
        print(f"❌ RealSense 카메라 연결 실패: {e}")
        return None


def get_frame_realsense(pipeline: rs.pipeline) -> Optional[np.ndarray]:
    """RealSense에서 컬러 프레임 가져오기"""
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            return np.asanyarray(color_frame.get_data())
    except Exception:
        pass
    return None


def main() -> None:
    args = parse_args()

    resolved_device = resolve_device_argument(args.device)
    if resolved_device != args.device:
        print(f"⚙️  추론 장치 자동 선택: {resolved_device}")
    args.device = resolved_device

    gui_enabled = args.show and is_gui_available()
    if args.show and not gui_enabled:
        print("⚠️  GUI 환경(DESKTOP/DISPLAY)을 찾을 수 없어 imshow를 사용할 수 없습니다.")
        print("    로컬 환경 또는 X11 포워딩을 사용하거나 --show 옵션을 제거하세요.")

    model = YOLO(args.weights)
    H = load_homography(args.homography)

    # RealSense 카메라 사용 여부 확인
    use_realsense = args.source.lower() in ["realsense", "rs", "d435i"]
    
    pipeline = None
    cap = None
    
    if use_realsense:
        pipeline = init_realsense()
        if pipeline is None:
            print("⚠️  RealSense 카메라를 사용할 수 없습니다. 웹캠으로 전환합니다.")
            use_realsense = False
            source = 0
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("❌ 웹캠도 열 수 없습니다.")
                return
    else:
        source: Union[int, str] = to_int_if_digit(args.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("❌ 입력 소스를 열 수 없습니다. 인덱스/경로/URL을 확인하세요.")
            return

    writer = None
    if args.save:
        # 캡처 정보가 없는 경우 대비해 첫 프레임에서 실제 크기로 초기화
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 임시 크기, 첫 프레임에서 갱신
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, (640, 480))

    prev_time = time.time()
    initialized_size = False
    frame_idx = 0
    window_created = False

    try:
        while True:
            if use_realsense and pipeline:
                frame = get_frame_realsense(pipeline)
                ok = frame is not None
            else:
                ok, frame = cap.read()
            
            if not ok or frame is None:
                break
            frame_idx += 1

            if writer is not None and not initialized_size:
                h, w = frame.shape[:2]
                # 재초기화
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 30.0, (w, h))
                initialized_size = True

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
                stream=False,
            )
            annotated = results[0].plot()

            # 콘솔 로그: 탐지 수/FPS + 클래스별 박스 중점 좌표
            if args.log_every and frame_idx % max(args.log_every, 1) == 0:
                num_det = 0
                centers_by_class = {}
                bev_by_class = {}
                try:
                    boxes = results[0].boxes
                    if boxes is not None and hasattr(boxes, "xyxy"):
                        xyxy = boxes.xyxy
                        cls_idx = getattr(boxes, "cls", None)
                        if xyxy is not None:
                            xyxy_np = xyxy.detach().cpu().numpy()
                            num_det = int(xyxy_np.shape[0])
                        names_map = getattr(results[0], "names", None) or {}
                        if cls_idx is not None:
                            cls_np = cls_idx.detach().int().cpu().numpy().reshape(-1)
                        else:
                            cls_np = []
                        if num_det > 0:
                            for i in range(num_det):
                                x1, y1, x2, y2 = xyxy_np[i]
                                cx = (x1 + x2) / 2.0
                                cy = (y1 + y2) / 2.0
                                label_idx = int(cls_np[i]) if len(cls_np) == num_det else -1
                                label_name = names_map.get(label_idx, str(label_idx))
                                centers_by_class.setdefault(label_name, []).append((float(cx), float(cy)))
                                if H is not None:
                                    bx = (x1 + x2) / 2.0
                                    by = y2
                                    bev = apply_homography_point(H, bx, by)
                                    if bev is not None:
                                        bev_by_class.setdefault(label_name, []).append(bev)
                except Exception:
                    pass
                now = time.time()
                fps_curr = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                print(f"[{frame_idx}] det={num_det}  fps={fps_curr:.1f}")
                if centers_by_class:
                    for cls_name, centers in centers_by_class.items():
                        coords = ", ".join(f"({c[0]:.1f},{c[1]:.1f})" for c in centers)
                        print(f"  - {cls_name}: {coords}")
                if bev_by_class:
                    for cls_name, pts in bev_by_class.items():
                        coords = ", ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in pts)
                        print(f"    BEV {cls_name}: {coords}")

            if args.fps:
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(annotated)

            if gui_enabled:
                if not window_created:
                    try:
                        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                        window_created = True
                    except cv2.error as exc:
                        print(f"❌ OpenCV 윈도우 생성 실패: {exc}")
                        print("   DISPLAY 설정을 확인하거나 --show 옵션을 비활성화하세요.")
                        gui_enabled = False
                        continue
                try:
                    cv2.imshow(WINDOW_TITLE, annotated)
                except cv2.error as exc:
                    print(f"❌ OpenCV imshow 오류: {exc}")
                    print("   GUI 환경 문제로 imshow가 비활성화됩니다.")
                    gui_enabled = False
                else:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if window_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


