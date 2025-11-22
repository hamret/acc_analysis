import numpy as np


class SyncCalibrator:

    # ------------------------------------------------------------
    # 1) YOLO 속도 vs 텔레메트리 속도 자동 싱크
    # ------------------------------------------------------------
    def auto_sync(self, yolo_speed, tel_speed):
        """
        비디오 기반 YOLO 속도데이터 vs 텔레메트리 속도데이터 상관 분석으로 offset 찾기
        """

        print("[SYNC] 자동 싱크 계산중...")

        # 리스트 → numpy (NaN 자동 정리)
        y = np.array(yolo_speed, dtype=float)
        t = np.array(tel_speed, dtype=float)

        # NaN 제거
        y = np.nan_to_num(y)
        t = np.nan_to_num(t)

        # 길이 맞추기
        min_len = min(len(y), len(t))
        y = y[:min_len]
        t = t[:min_len]

        if min_len < 10:
            print("[SYNC] 데이터가 너무 짧아서 싱크 불가. offset=0 적용")
            return 0

        # 상관계수 계산
        corr = np.correlate(y - y.mean(), t - t.mean(), mode="full")
        offset_idx = corr.argmax() - (len(y) - 1)

        print(f"[SYNC] 계산된 offset (프레임) = {offset_idx}")
        return int(offset_idx) * 16  # ms 환산 (예: 60fps 기준 16ms/frame)

    # ------------------------------------------------------------
    # 2) frame → telemetry index 매핑
    # ------------------------------------------------------------
    def generate_frame_map(self, fps, n_video, n_tel, sync_offset):
        """
        fps : 영상 FPS
        n_video : 전체 영상 프레임 수
        n_tel : 텔레메트리 샘플 수
        sync_offset : ms 단위 싱크 오프셋 (auto_sync 결과)
        """

        ms_per_frame = 1000 / fps
        frame_map = []

        for f in range(n_video):
            # f 프레임의 timestamp (ms)
            t = f * ms_per_frame - sync_offset

            # 텔레메트리 인덱스
            tel_idx = int(t / ms_per_frame)

            # 범위 밖이면 None
            if tel_idx < 0 or tel_idx >= n_tel:
                frame_map.append(None)
            else:
                frame_map.append(tel_idx)

        return frame_map
