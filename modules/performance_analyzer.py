import numpy as np


class PerformanceAnalyzer:
    def analyze(self, telemetry, trajectory):
        speed = telemetry["speed"].values
        throttle = telemetry["throttle"].values
        brake = telemetry["brake"].values
        steer = telemetry["steerangle"].values

        return {
            "max_speed": float(np.max(speed)),
            "avg_speed": float(np.mean(speed)),
            "avg_throttle": float(np.mean(throttle)),
            "avg_brake": float(np.mean(brake)),
            "avg_steer": float(np.mean(np.abs(steer))),
        }
