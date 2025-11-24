import numpy as np
import scipy.signal as signal


class SyncCalibrator:

    # =========================================================
    # 1) YOLO speed 계산 (x, y 픽셀 이동 기반)
    # =========================================================
    def compute_yolo_speed(self, car_pos):
        speeds = []
        prev = None
        for p in car_pos:
            if prev is None or p is None:
                speeds.append(0)
                prev = p
                continue

            px = np.linalg.norm(np.array(p) - np.array(prev))
            speeds.append(px)
            prev = p
        return np.array(speeds)

    # =========================================================
    # 2) smoothing 적용 (30프레임 윈도우)
    # =========================================================
    def smooth(self, arr, win=30):
        if len(arr) < win:
            return arr
        kernel = np.ones(win) / win
        return np.convolve(arr, kernel, mode="same")

    # =========================================================
    # 3) 신호 정규화
    # =========================================================
    def normalize(self, x):
        x = np.array(x, dtype=float)
        x = x - np.nanmean(x)
        std = np.nanstd(x)
        if std < 1e-6:
            std = 1.0
        return x / std

    # =========================================================
    # 4) Cross-correlation 기반 자동 sync
    # =========================================================
    def auto_sync(self, yolo_speed_raw, telemetry_speed_raw):

        # 1) YOLO speed smoothing
        yolo_speed = self.smooth(self.normalize(yolo_speed_raw), win=30)

        # 2) 텔레메트리 speed smoothing
        tel_speed = self.smooth(self.normalize(telemetry_speed_raw), win=200)

        # 3) 길이 너무 짧으면 abort
        if len(yolo_speed) < 100 or len(tel_speed) < 100:
            print("[SYNC] 신호 길이가 너무 짧아서 sync=0 반환")
            return 0

        print(f"[SYNC] YOLO len = {len(yolo_speed)}, TEL len = {len(tel_speed)}")

        # 4) cross-correlation 계산
        corr = signal.correlate(tel_speed, yolo_speed, mode="full")

        # peak 찾기
        shift = np.argmax(corr) - (len(yolo_speed) - 1)

        print(f"[SYNC] best offset (frame) = {shift}")
        return shift

    # =========================================================
    # 5) frame map 생성
    # =========================================================
    def generate_frame_map(self, fps, n_video, n_tel, sync_offset):
        frame_map = []

        for i in range(n_video):
            tel_idx = i + sync_offset
            if 0 <= tel_idx < n_tel:
                frame_map.append(tel_idx)
            else:
                frame_map.append(None)

        return frame_map

    def generate_frame_map_by_distance(self, yolo_dist, tel_dist):
        yolo_dist = np.array(yolo_dist, dtype=float)
        tel_dist = np.array(tel_dist, dtype=float)

        # 1) Normalize to 0~1 progression
        y = (yolo_dist - yolo_dist.min()) / (yolo_dist.max() - yolo_dist.min() + 1e-6)
        t = (tel_dist - tel_dist.min()) / (tel_dist.max() - tel_dist.min() + 1e-6)

        frame_map = []

        j = 0  # telemetry index
        T = len(t)

        # 2) For each YOLO point, find nearest telemetry progression
        for i in range(len(y)):
            yi = y[i]

            # advance telemetry index until t[j] >= yi
            while j + 1 < T and abs(t[j + 1] - yi) < abs(t[j] - yi):
                j += 1

            frame_map.append(j)

        return frame_map
