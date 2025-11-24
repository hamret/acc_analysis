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
        dist_raw = telemetry["distance"].to_numpy()  # ← Spa 0~7000m 실측값 그대로

        # =======================================
        # 1) 실제 heading 계산 (deg/s → rad/s)
        # =======================================
        dt = np.diff(time, prepend=time[0])
        dt = np.clip(dt, 0.001, 0.2)

        yaw_rate_rad = np.radians(yaw_rate_deg)
        heading = np.cumsum(yaw_rate_rad * dt)

        # =======================================
        # 2) XY는 distance 를 기준으로 정확하게 생성
        # =======================================
        x = dist_raw * np.cos(heading)
        y = dist_raw * np.sin(heading)

        trajectory = {
            "x": x.tolist(),
            "y": y.tolist(),
            "heading": heading.tolist(),
            "speed": speed_kmh.tolist(),
            "distance": dist_raw.tolist()
        }

        print("[ANALYZE] 텔레메트리 XY 변환 완료!")
        return trajectory

    # YOLO distance → 누적거리 계산
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
            total += d           # 🔥 누적
            dist.append(total)
            last = p

        return dist
