import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import WorldModel
from filterpy.kalman import KalmanFilter

torch.serialization.add_safe_globals([WorldModel])


class VideoProcessor:

    def __init__(self):
        print("[VideoProcessor] 초기화 중...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VideoProcessor] YOLO 디바이스: {self.device}")

        # -------------------------------------------------------
        # 1) YOLO 모델 자동 로드
        # -------------------------------------------------------
        model_candidates = [
            "models/yolov8x-worldv2.pt",
            "models/yolov8m.pt",
            "yolov8x-worldv2.pt",
            "yolov8m.pt",
        ]

        model_path = None
        for m in model_candidates:
            if os.path.exists(m):
                model_path = m
                break

        if model_path is None:
            raise FileNotFoundError("YOLO 모델 파일을 찾을 수 없습니다.")

        print(f"[VideoProcessor] 모델 로딩: {model_path}")
        self.model = YOLO(model_path)

        # -------------------------------------------------------
        # 2) 차량 HSV 히스토그램 (사용자가 제공)
        # -------------------------------------------------------
        self.hist_h = np.load(r"C:\Users\user\PycharmProjects\acc_analysis\car_hist_h.npy")
        self.hist_s = np.load(r"C:\Users\user\PycharmProjects\acc_analysis\car_hist_s.npy")
        self.hist_v = np.load(r"C:\Users\user\PycharmProjects\acc_analysis\car_hist_v.npy")

        # Resize → 모든 hist 크기 동일하게 보정
        self.hist_h = cv2.resize(self.hist_h.astype(np.float32), (1, 180)).flatten()
        self.hist_s = cv2.resize(self.hist_s.astype(np.float32), (1, 256)).flatten()
        self.hist_v = cv2.resize(self.hist_v.astype(np.float32), (1, 256)).flatten()

        # -------------------------------------------------------
        # 3) Kalman Filter
        # -------------------------------------------------------
        self.kf = self._create_kalman()
        self.kf_ready = False

        self.last_box = None
        self.alpha = 0.65  # Smooth factor


    # ===========================================================
    # Kalman 필터 구성
    # ===========================================================
    def _create_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)

        kf.x = np.array([0, 0, 0, 0])

        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        kf.P *= 15
        kf.R = np.eye(2) * 3
        kf.Q = np.eye(4) * 0.05
        return kf


    # ===========================================================
    # Bhattacharyya 거리 계산
    # ===========================================================
    def bhatta_distance(self, h1, h2):
        h1 = h1.flatten().astype(np.float64)
        h2 = h2.flatten().astype(np.float64)

        h1 /= (h1.sum() + 1e-6)
        h2 /= (h2.sum() + 1e-6)

        bc = np.sum(np.sqrt(h1 * h2))
        return np.sqrt(max(0, 1 - bc))


    # ===========================================================
    # 후보 박스 중에서 "가장 우리 차량" 선택
    # ===========================================================
    def choose_best_box(self, frame, boxes):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_score = 9999
        best_box = None

        for b in boxes:
            x1, y1, x2, y2 = map(int, b)

            # 너무 작으면 제외
            if x2 - x1 < 40 or y2 - y1 < 40:
                continue

            crop = hsv[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            h = cv2.calcHist([crop], [0], None, [180], [0, 180])
            s = cv2.calcHist([crop], [1], None, [256], [0, 256])
            v = cv2.calcHist([crop], [2], None, [256], [0, 256])

            score = (
                self.bhatta_distance(self.hist_h, h) +
                self.bhatta_distance(self.hist_s, s) +
                self.bhatta_distance(self.hist_v, v)
            )

            if score < best_score:
                best_score = score
                best_box = b

        return best_box


    # ===========================================================
    # YOLO + HSV 히스토그램 기반 차량 인식
    # ===========================================================
    def detect_car(self, frame):
        H, W = frame.shape[:2]

        results = self.model.predict(
            frame, conf=0.35, iou=0.45,
            device=self.device, verbose=False
        )

        if len(results) == 0 or results[0].boxes is None:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()

        # ROI: 화면 아래 45%만 허용
        candidates = []
        for b in boxes:
            x1, y1, x2, y2 = b
            cy = (y1 + y2) / 2
            if cy > H * 0.45:
                candidates.append(b)

        if len(candidates) == 0:
            return None

        # 히스토그램 기반으로 "우리 차"와 가장 가까운 박스 선택
        best = self.choose_best_box(frame, candidates)
        if best is None:
            return None

        # Smooth
        best = np.array(best)
        if self.last_box is not None:
            best = self.last_box * self.alpha + best * (1 - self.alpha)
        self.last_box = best.copy()

        return best.astype(int)


    # ===========================================================
    # 영상 처리
    # ===========================================================
    def process(self, video_path):
        print("[VideoProcessor] 비디오 처리 시작...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        traj = {"car_pos": [], "speed": []}

        last_center = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            box = self.detect_car(frame)

            if box is not None:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not self.kf_ready:
                    self.kf.x[:2] = [cx, cy]
                    self.kf_ready = True

                self.kf.update([cx, cy])

            else:
                if self.kf_ready:
                    self.kf.predict()
                else:
                    traj["car_pos"].append(None)
                    traj["speed"].append(0)
                    continue

            # 칼만 예측 항상 실행
            self.kf.predict()
            cx, cy = int(self.kf.x[0]), int(self.kf.x[1])

            # speed 계산
            if last_center is not None:
                spd = np.linalg.norm(np.array([cx, cy]) - np.array(last_center))
            else:
                spd = 0

            last_center = (cx, cy)

            traj["car_pos"].append((cx, cy))
            traj["speed"].append(spd)

        cap.release()

        print("[VideoProcessor] 영상 처리 완료!")
        return {
            "fps": fps,
            "width": W,
            "height": H,
            "frames": total
        }, traj


    # ===========================================================
    # Warp된 레이싱 라인 + 차량 위치를 영상에 그리기
    # ===========================================================
    def render_overlay(self, input_video, warped, yolo_traj, output_path):
        print("[VideoProcessor] 오버레이 렌더링 시작...")

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 차량 위치 표시
            if idx < len(yolo_traj["car_pos"]) and yolo_traj["car_pos"][idx] is not None:
                cx, cy = yolo_traj["car_pos"][idx]
                cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

            # 레이싱 라인 현재 프레임
            if idx < len(warped) and warped[idx] is not None:
                u, v = warped[idx]
                cv2.circle(frame, (u, v), 5, (0, 128, 255), -1)

            # 과거 라인
            for j in range(max(0, idx - 250), idx):
                if j < len(warped) and warped[j] is not None:
                    uu, vv = warped[j]
                    cv2.circle(frame, (uu, vv), 2, (0, 80, 255), -1)

            out.write(frame)
            idx += 1

        cap.release()
        out.release()

        print("[VideoProcessor] 오버레이 렌더링 완료!")
        print(f"[VideoProcessor] → {output_path}")
