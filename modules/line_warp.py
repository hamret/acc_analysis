import numpy as np


class LineWarpEngine:

    def __init__(self):
        # 화면 좌표로 보내기 위한 임의 스케일/오프셋 (튜닝 포인트)
        self.scale_x = 0.035
        self.scale_y = 22.0

        self.offset_x = 0.52
        self.offset_y = 0.82

    def world_to_screen(self, x, y, meta):
        """트랙 좌표(x, y)를 영상 좌표(u, v)로 선형 매핑."""
        W = meta["width"]
        H = meta["height"]

        u = int(W * self.offset_x + y * self.scale_y)
        v = int(H * self.offset_y - x * self.scale_x)

        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))
        return u, v

    def warp(self, trajectory, meta, yolo_pos):
        """
        frame_map : [frame_idx -> telemetry_idx or None]
        각 프레임마다 real / ideal 좌표를 화면 픽셀로 변환.
        """
        xs = trajectory["x"]
        ys = trajectory["y"]
        ideal_x = trajectory["ideal_x"]
        ideal_y = trajectory["ideal_y"]
        fm = trajectory["frame_map"]

        n_frames = min(len(fm), len(yolo_pos))  # 영상 프레임 수 기준
        warped_real = []
        warped_ideal = []

        for i in range(n_frames):
            tel_idx = fm[i]
            if tel_idx is None:
                warped_real.append(None)
                warped_ideal.append(None)
                continue

            # 텔레 인덱스로 real world 좌표 선택
            xr = xs[tel_idx]
            yr = ys[tel_idx]
            ui, vi = self.world_to_screen(xr, yr, meta)
            warped_real.append((ui, vi))

            # ideal line도 동일한 텔레 인덱스 기준으로 매핑
            xi = ideal_x[tel_idx]
            yi = ideal_y[tel_idx]
            # ideal_x, ideal_y가 이미 화면 좌표면 그대로 사용,
            # world 좌표면 위와 같이 world_to_screen을 한 번 더 태워도 됨.
            warped_ideal.append((int(xi), int(yi)))

        return warped_real, warped_ideal
