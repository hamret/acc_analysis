import numpy as np


class SyncCalibrator:
    def auto_sync(self, yolo_speed, tel_speed):
        """
        비디오 속도 신호와 텔레메트리 속도의 유사도를 기반으로 자동 싱크 계산
        """
        print("[SYNC] 자동 싱크 계산...")

        y = np.array(yolo_speed)
        t = np.array(tel_speed)

        min_len = min(len(y), len(t))
        y = y[:min_len]
        t = t[:min_len]

        # 단순 상관계수 기반
        corr = np.correlate(y - y.mean(), t - t.mean(), mode="full")
        offset = corr.argmax() - (len(y) - 1)

        print(f"[SYNC] 계산된 offset = {offset} frames")
        return offset * 16  # ms 추정

    def generate_frame_map(self, fps, n_video, n_tel, sync_offset):
        ms_per_frame = 1000 / fps

        frame_map = []
        for f in range(n_video):
            t = f * ms_per_frame - sync_offset
            tel_idx = int(t / ms_per_frame)
            if tel_idx < 0 or tel_idx >= n_tel:
                frame_map.append(None)
            else:
                frame_map.append(tel_idx)

        return frame_map
