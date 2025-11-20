import cv2
import numpy as np
from ultralytics import YOLO
import torch


class VideoProcessor:
    def __init__(self):
        # YOLO 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VideoProcessor] YOLO 디바이스: {self.device}")

        self.model = YOLO("models/yolov8m.pt")

    def process(self, video_path):
        """
        YOLO + ByteTrack 기반 차량 추적
        """
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        yolo_speed = []
        yolo_positions = []

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)

            # 차량 1대만 있는 영상 기준
            best_box = None
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 2:  # 자동차 클래스
                        best_box = box.xywh[0].cpu().numpy()
                        break

            if best_box is None:
                yolo_positions.append((None, None))
                yolo_speed.append(0)
            else:
                x, y, w, h = best_box
                yolo_positions.append((x, y))

                # 간단한 속도 추정
                if frame_idx > 0:
                    prev_x, prev_y = yolo_positions[-2]
                    if prev_x is not None:
                        speed = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        yolo_speed.append(speed)
                    else:
                        yolo_speed.append(0)
                else:
                    yolo_speed.append(0)

            if frame_idx % 100 == 0:
                print(f"YOLO 진행률: {frame_idx}/{total_frames}")

            frame_idx += 1

        cap.release()

        meta = {
            "fps": fps,
            "total_frames": total_frames,
            "video_path": video_path,
        }

        yolo_traj = {
            "positions": yolo_positions,
            "speed": yolo_speed
        }

        return meta, yolo_traj

    def render_overlay(self, video_path, warped_lines, traj, output_path):
        """
        헬리캠 영상에 레이싱 라인 오버레이.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # warped_lines는 frame map 기준
            if frame_idx < len(warped_lines):
                pts = warped_lines[frame_idx]
                for i in range(1, len(pts)):
                    cv2.line(frame,
                             tuple(map(int, pts[i - 1])),
                             tuple(map(int, pts[i])),
                             (255, 80, 0), 4)

            writer.write(frame)
            frame_idx += 1

        writer.release()
        cap.release()

        print(f"[VideoProcessor] 오버레이 렌더링 완료: {output_path}")
