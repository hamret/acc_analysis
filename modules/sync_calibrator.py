import numpy as np
import scipy.signal as signal


class SyncCalibrator:

    def compute_yolo_speed(self, car_pos):
        """프레임별 bbox 중심 이동량으로 대략적인 속도 시퀀스 생성."""
        speed = []
        prev = None

        for p in car_pos:
            if prev is None or p is None:
                speed.append(0.0)
                prev = p
                continue
            px = np.linalg.norm(np.array(p) - np.array(prev))
            speed.append(float(px))
            prev = p

        return np.array(speed, dtype=float)

    def normalize(self, x):
        x = np.array(x, dtype=float)
        if x.size == 0:
            return x
        return (x - x.mean()) / (x.std() + 1e-6)

    def auto_sync_speed(self, yolo_speed, tel_speed):
        """YOLO 속도 시퀀스와 텔레 속도의 cross-correlation으로 offset 찾기."""
        if len(yolo_speed) == 0 or len(tel_speed) == 0:
            print("[SYNC] 빈 속도 시퀀스 → offset=0 사용")
            return 0

        y = self.normalize(yolo_speed)
        t = self.normalize(tel_speed)

        corr = signal.correlate(t, y, mode="full")
        shift = int(np.argmax(corr) - (len(y) - 1))

        print(f"[SYNC] offset = {shift}")
        return shift

    def generate_frame_map(self, n_video, n_tel, offset):
        """
        frame i  -> telemetry index (또는 None)
        """
        frame_map = []
        for i in range(n_video):
            idx = i + offset
            if 0 <= idx < n_tel:
                frame_map.append(idx)
            else:
                frame_map.append(None)
        return frame_map
