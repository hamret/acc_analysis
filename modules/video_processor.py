import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[VideoProcessor] ultralytics가 설치되어 있지 않습니다. YOLO 트래킹을 사용할 수 없습니다.")


class VideoProcessor:

    def __init__(self, model_path="models/yolov8x-worldv2.pt", device="cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None

        if YOLO is not None:
            try:
                self.model = YOLO(self.model_path)
                print(f"[VideoProcessor] YOLO 모델 로딩: {self.model_path}")
            except Exception as e:
                print("[VideoProcessor] YOLO 모델 로딩 실패:", repr(e))
                self.model = None

    def process(self, video_path):
        """
        영상 메타데이터 + YOLO 기반 car_pos 시퀀스 생성.
        car_pos[i] = (cx, cy) or None
        """
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        meta = {"width": W, "height": H, "fps": fps}

        # YOLO가 없으면 car_pos를 전부 None으로 채워서라도 길이는 맞춰줌
        if self.model is None:
            print("[VideoProcessor] YOLO 미사용 → car_pos를 None으로 채웁니다.")
            cap = cv2.VideoCapture(video_path)
            car_pos = []
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                car_pos.append(None)
            cap.release()
            return meta, {"car_pos": car_pos}

        # ultralytics YOLO tracking
        car_pos = []
        print("[VideoProcessor] YOLO tracking 시작...")
        results = self.model.track(
            source=video_path,
            stream=True,
            device=self.device,
            verbose=False,
            persist=True,
            conf=0.4
        )

        for r in results:
            # r.orig_shape, r.boxes 등 사용 가능
            if r.boxes is None or len(r.boxes) == 0:
                car_pos.append(None)
                continue

            # 가장 큰 bbox를 "내 차"라고 가정
            boxes = r.boxes.xyxy
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx_max = int(areas.argmax().item())
            x1, y1, x2, y2 = boxes[idx_max].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            car_pos.append((cx, cy))

        print(f"[VideoProcessor] YOLO tracking 완료. 프레임 수: {len(car_pos)}")

        return meta, {"car_pos": car_pos}

    def render_overlay(self, video_path, warped_real, warped_ideal, yolo_traj, outpath):
        """
        - warped_real: 각 프레임별 real line 위치 (u, v) 또는 None
        - warped_ideal: 각 프레임별 ideal line 위치 (u, v) 또는 None
        - yolo_traj["car_pos"]: YOLO가 잡은 차량 위치
        """
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(outpath, fourcc, fps, (W, H))

        car_pos = yolo_traj.get("car_pos", [])

        # ideal line은 전체 궤적을 하나의 polyline으로 항상 보여줌
        ideal_points = [p for p in warped_ideal if p is not None]
        ideal_poly = None
        if len(ideal_points) >= 2:
            ideal_poly = np.array(ideal_points, dtype=np.int32).reshape(-1, 1, 2)

        real_trail = []  # 지금까지 지나온 real 궤적

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # index 범위 체크
            if idx < len(warped_real):
                if warped_real[idx] is not None:
                    real_trail.append(warped_real[idx])

            # -------------------------
            # ideal line 전체 (녹색)
            # -------------------------
            if ideal_poly is not None:
                cv2.polylines(frame, [ideal_poly], False, (0, 255, 0), 2)

            # -------------------------
            # 지금까지의 real line (파란색, 차량 뒤로 길게 남음)
            # -------------------------
            if len(real_trail) >= 2:
                rt = np.array(real_trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [rt], False, (255, 0, 0), 2)

            # -------------------------
            # YOLO car marker (빨강 점)
            # -------------------------
            if idx < len(car_pos):
                pos = car_pos[idx]
                if pos is not None:
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 6, (0, 0, 255), -1)

            out.write(frame)
            idx += 1

        out.release()
        cap.release()
        print(f"[VideoProcessor] overlay 영상 저장 완료: {outpath}")
