import numpy as np

class TrajectoryAnalyzer:

    def create_trajectory(self, telemetry):
        print("[ANALYZE] 텔레메트리 XY 변환 시작...")

        required = ["time", "roty", "speed", "distance"]
        for col in required:
            if col not in telemetry.columns:
                raise RuntimeError(f"'{col}' 컬럼이 텔레메트리에 없습니다.")

        time = telemetry["time"].to_numpy()
        yaw_rate_deg = telemetry["roty"].to_numpy()
        speed_kmh = telemetry["speed"].to_numpy()
        dist_raw = telemetry["distance"].to_numpy()  # Spa 0~19855m 실측값

        # =======================================
        # 1) heading 계산
        # =======================================
        dt = np.diff(time, prepend=time[0])
        dt = np.clip(dt, 0.001, 0.2)

        yaw_rate_rad = np.radians(yaw_rate_deg)
        heading = np.cumsum(yaw_rate_rad * dt)

        # =======================================
        # 2) XY 생성 (distance 기준)
        # =======================================
        x = dist_raw * np.cos(heading)
        y = dist_raw * np.sin(heading)

        # =======================================
        # 🔥 3) 0~1m 구간 제거 (초반 완전 정지 구간 삭제)
        # =======================================
        dist = dist_raw
        valid_start = np.argmax(dist > 1.0)   # distance가 1m 넘는 지점부터

        x = x[valid_start:]
        y = y[valid_start:]
        heading = heading[valid_start:]
        speed_kmh = speed_kmh[valid_start:]
        dist_raw = dist_raw[valid_start:]

        trajectory = {
            "x": x.tolist(),
            "y": y.tolist(),
            "heading": heading.tolist(),
            "speed": speed_kmh.tolist(),
            "distance": dist_raw.tolist()
        }

        print("[ANALYZE] 텔레메트리 XY 변환 완료!")
        return trajectory


    # ===========================================
    # YOLO distance → 누적거리 계산
    # ===========================================
    def create_yolo_distance(self, car_pos):
        dist = []
        last = None
        total = 0

        for p in car_pos:
            if p is None or last is None:
                dist.append(total)
                last = p
                continue

            d = np.linalg.norm(np.array(p) - np.array(last))
            total += d
            dist.append(total)
            last = p

        return dist
