import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import WorldModel

# PyTorch 2.6 safe-load fix
torch.serialization.add_safe_globals([WorldModel])

class VideoProcessor:
    def __init__(self):
        print("[VideoProcessor] 초기화 중...")

        # ------------------------------------------------------------
        # 1) GPU 장치 선택
        # ------------------------------------------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VideoProcessor] YOLO 디바이스: {self.device}")

        # ------------------------------------------------------------
        # 2) 모델 자동 선택
        # ------------------------------------------------------------
        # 우선순위:
        # 1) 프로젝트 로컬에 worldv2 있으면 그걸 사용
        # 2) 없으면 기존 yolov8m.pt 사용
        # ------------------------------------------------------------
        model_candidates = [
            "models/yolov8x-worldv2.pt",
            "models/yolov8m.pt",
            "yolov8x-worldv2.pt",  # pip 설치된 모델
            "yolov8m.pt"
        ]

        model_path = None
        for m in model_candidates:
            if os.path.exists(m):
                model_path = m
                break

        if model_path is None:
            raise FileNotFoundError(
                "YOLO 모델 파일을 찾을 수 없습니다.\n"
                "models/yolov8x-worldv2.pt 또는 yolov8m.pt 위치에 모델을 넣어주세요."
            )

        print(f"[VideoProcessor] 모델 로딩: {model_path}")
        self.model = YOLO(model_path)

        # ------------------------------------------------------------
        # 3) 검출 안정화를 위한 버퍼
        # ------------------------------------------------------------
        self.last_box = None   # 마지막 박스
        self.smooth_factor = 0.7  # 0 ~ 1 (값 커질수록 더 부드러움)

    # ------------------------------------------------------------------
    # YOLO 박스 안정화
    # ------------------------------------------------------------------
    def smooth_box(self, new_box):
        if self.last_box is None:
            self.last_box = new_box
            return new_box

        # previous * a + new * (1-a)
        smoothed = self.last_box * self.smooth_factor + new_box * (1 - self.smooth_factor)
        self.last_box = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # 차량 검출 (YOLO)
    # ------------------------------------------------------------------
    def detect_car(self, frame):
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=0.3,       # 인식 정확도 강화
            iou=0.5,
            verbose=False,
            device=self.device
        )

        if len(results) == 0:
            return None

        r = results[0]
        if r.boxes is None or len(r.boxes.xyxy) == 0:
            return None

        # 가장 큰 박스 = 차량일 확률 높음
        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_idx = np.argmax(areas)
        box = boxes[main_idx]

        # 안정화 적용
        box = self.smooth_box(box)

        x1, y1, x2, y2 = box.astype(int)
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # 비디오 처리 + YOLO 검출 + 차량 위치 + 속도 추정
    # ------------------------------------------------------------------
    def process(self, video_path):
        print("[VideoProcessor] 비디오 처리 시작...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[VideoProcessor] 영상 로드 실패: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VideoProcessor] FPS={fps}, Size={width}x{height}, Frames={total_frames}")

        trajectory = {
            "speed": [],
            "car_pos": []
        }

        frame_idx = 0
        last_center = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 차량 검출
            box = self.detect_car(frame)

            if box:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                trajectory["car_pos"].append((cx, cy))

                # 영상 속도 추정 = center 이동량
                if last_center is not None:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array(last_center))
                else:
                    dist = 0

                trajectory["speed"].append(dist)
                last_center = (cx, cy)

            else:
                trajectory["car_pos"].append(None)
                trajectory["speed"].append(0)
                last_center = None

            frame_idx += 1

        cap.release()

        print("[VideoProcessor] 영상 처리 완료!")
        return \
            {"fps": fps, "width": width, "height": height, "frames": total_frames}, \
            trajectory

    # ------------------------------------------------------------------
    # Warp된 라인을 영상 위에 그려서 MP4로 저장
    # ------------------------------------------------------------------
    def render_overlay(self, input_video, warped_points, yolo_traj, output_path):
        print("[VideoProcessor] 오버레이 렌더링 시작...")

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 박스 + 차량 경로 표시
            if yolo_traj["car_pos"][idx] is not None:
                cx, cy = yolo_traj["car_pos"][idx]
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # 레이싱 라인 표시
            if warped_points[idx] is not None:
                u, v = warped_points[idx]
                cv2.circle(frame, (u, v), 4, (0, 128, 255), -1)

            # 누적 레이싱 라인
            for i in range(max(0, idx - 300), idx):
                if warped_points[i] is not None:
                    uu, vv = warped_points[i]
                    cv2.circle(frame, (uu, vv), 1, (0, 80, 255), -1)

            out.write(frame)
            idx += 1

        cap.release()
        out.release()

        print("[VideoProcessor] 오버레이 렌더링 완료!")
        print(f"[VideoProcessor] 출력 파일 → {output_path}")
