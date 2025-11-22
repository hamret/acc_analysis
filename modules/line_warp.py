import numpy as np
import cv2


class LineWarpEngine:
    """
    텔레메트리에서 얻은 XY 레이싱 라인을
    드론/헬리캠 영상 좌표계로 투영하는 엔진
    """

    def __init__(self):
        # 기본적인 전방 카메라 FOV 세팅 (ACC 기준 근사치)
        self.fov_x = 78     # degrees
        self.fov_y = 42     # degrees

    # ------------------------------------------------------------
    # 1) 월드 좌표계 -> 카메라 좌표계 (간단한 시점 변환)
    # ------------------------------------------------------------
    def world_to_camera(self, x, y, meta):
        """
        world XY -> 영상 좌표(u, v)
        meta 안에는 영상폭/높이, 시점, scale 등이 들어있음
        """

        W, H = meta["width"], meta["height"]

        # 텔레메트리 XY는 스파 서킷 평면 좌표인데,
        # 영상에는 차량이 아래 중앙에 있으므로 기준 이동
        cx, cy = 0, 0  # 차량 위치(원점)

        # 상대 위치
        dx = x - cx
        dy = y - cy

        # 헬리캠의 약간 위쪽에서 내려보는 시점
        cam_height = 30.0
        dz = cam_height

        # 카메라 투영
        fx = W / (2 * np.tan(np.radians(self.fov_x / 2)))
        fy = H / (2 * np.tan(np.radians(self.fov_y / 2)))

        # 회전 없이 단순 투영
        u = fx * (dx / dz) + (W / 2)
        v = fy * (dy / dz) + (H * 0.8)  # 차량이 하단 20% 위치

        return int(u), int(v)

    # ------------------------------------------------------------
    # 2) 전체 레이싱 라인을 드론 영상 좌표계로 변환
    # ------------------------------------------------------------
    def warp_lines_to_video_view(self, trajectory, meta):
        """
        trajectory = {
            "x": [...],
            "y": [...],
            "frame_map": [...]
        }
        meta = {
            "width": int,
            "height": int,
            "fps": float,
            ...
        }
        """

        print("[WARP] 레이싱 라인 영상 좌표 변환 시작...")

        xs = trajectory["x"]
        ys = trajectory["y"]
        fm = trajectory["frame_map"]

        warped = []

        for i in range(len(xs)):
            if fm[i] is None:
                warped.append(None)
                continue

            u, v = self.world_to_camera(xs[i], ys[i], meta)
            warped.append((u, v))

        print("[WARP] 좌표 변환 완료!")
        return warped

    # ------------------------------------------------------------
    # 3) 프레임에 맞춰 하이라이트 포인트 계산
    # ------------------------------------------------------------
    def get_highlight_point(self, warped_points, frame_idx):
        if frame_idx < 0 or frame_idx >= len(warped_points):
            return None
        return warped_points[frame_idx]
