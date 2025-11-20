import numpy as np


class LineWarpEngine:
    """
    헬리캠 영상에 텔레메트리 라인을 투영 (단순 스케일 기반)
    """

    def warp_lines_to_video_view(self, traj, meta):
        x = traj["x"]
        y = traj["y"]
        frame_map = traj["frame_map"]

        if frame_map is None:
            return []

        max_x = np.max(x)
        max_y = np.max(y)
        min_x = np.min(x)
        min_y = np.min(y)

        def normalize(v, minv, maxv):
            return (v - minv) / (maxv - minv + 1e-6)

        warped_frames = []

        for f in frame_map:
            if f is None:
                warped_frames.append([])
                continue

            if f >= len(x):
                warped_frames.append([])
                continue

            # 주변 20개 포인트만 표시
            pts = []
            for k in range(max(0, f - 20), min(len(x), f + 20)):
                nx = normalize(x[k], min_x, max_x)
                ny = normalize(y[k], min_y, max_y)

                # 화면 스케일
                px = int(nx * 1920)
                py = int(1080 - ny * 1080)
                pts.append((px, py))

            warped_frames.append(pts)

        return warped_frames
