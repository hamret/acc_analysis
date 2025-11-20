import numpy as np


class TrajectoryAnalyzer:
    def create_trajectory(self, telemetry):
        print("[ANALYZE] 텔레메트리 XY 변환…")

        distance = telemetry["distance"].to_numpy()
        speed = telemetry["speed"].to_numpy()
        steer = telemetry["steer"].to_numpy()

        heading = np.cumsum(steer * 0.00015)

        x = np.cumsum(np.cos(heading) * speed * 0.02)
        y = np.cumsum(np.sin(heading) * speed * 0.02)

        return {
            "distance": distance,
            "speed": speed,
            "steer": steer,
            "x": x,
            "y": y,
        }
