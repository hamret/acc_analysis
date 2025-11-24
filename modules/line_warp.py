import numpy as np
import cv2


class LineWarpEngine:
    """
    텔레메트리 XY 레이싱 라인을
    헬리캠(Top-down) 영상 좌표로 매핑하는 전용 엔진
    """

    def __init__(self):
        # 헬리캠 시야에 맞춘 스케일 기본값 (필요하면 동적으로 조정)
        self.scale_x = 0.035  # distance → 화면 이동량
        self.scale_y = 22.0  # steer/lateral → 좌우 이동량

        self.offset_x = 0.52  # 화면 중앙 기준 좌우 보정
        self.offset_y = 0.82  # 화면 아래쪽 위치 고정

    # ------------------------------------------------------------
    # world (telemetry) → screen (video)
    # ------------------------------------------------------------
    def world_to_screen(self, x, y, meta):
        """
        x: distance 기반 (앞뒤 이동)
        y: steer 누적 기반 (좌우 이동)
        """
        W = meta["width"]
        H = meta["height"]

        # 좌우 (y 축) → 화면 중심 기준 이동
        u = int(W * self.offset_x + y * self.scale_y)

        # 앞뒤 (x 축) → 아래쪽에서 위로 올라오게
        v = int(H * self.offset_y - x * self.scale_x)

        # 화면 밖으로 넘어가지 않도록 clamp
        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))

        return u, v

    # ------------------------------------------------------------
    # 텔레메트리 전체 라인을 화면 좌표로 매핑
    # ------------------------------------------------------------
    def warp_lines_to_video_view(self, trajectory, meta):
        print("[WARP] 레이싱 라인 영상 좌표 변환 시작...")

        xs = trajectory["x"]  # distance 기반
        ys = trajectory["y"]  # lateral 기반
        fm = trajectory["frame_map"]  # 프레임 매핑

        warped = []
        n = min(len(xs), len(fm))

        for i in range(n):
            if fm[i] is None:
                warped.append(None)
                continue

            u, v = self.world_to_screen(xs[i], ys[i], meta)
            warped.append((u, v))

        print("[WARP] 좌표 변환 완료!")
        # ===================== DEBUG OUTPUT ======================
        print("\n=== TRAJECTORY DEBUG ===")
        print("len x =", len(xs))
        print("x sample =", xs[:20])
        print("y sample =", ys[:20])
        print("frame_map sample =", fm[:20])

        print("len warped =", len(warped))
        print("warped sample =", warped[:20])
        print("=====================================================\n")
        # =========================================================

        return warped

    # ------------------------------------------------------------
    # 특정 프레임에 해당하는 warped 좌표 가져오기
    # ------------------------------------------------------------
    def get_highlight_point(self, warped_points, frame_idx):
        if frame_idx < 0 or frame_idx >= len(warped_points):
            return None
        return warped_points[frame_idx]
