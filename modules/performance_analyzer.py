import numpy as np

class PerformanceAnalyzer:

    def analyze(self, telemetry, trajectory):
        result = {}

        try:
            # =======================
            # 속도
            # =======================
            speed = telemetry.get("speed")
            if speed is not None:
                s = np.array(speed, dtype=float)
                result["avg_speed"] = float(s.mean())
                result["max_speed"] = float(s.max())
            else:
                result["avg_speed"] = 0
                result["max_speed"] = 0

            # =======================
            # 스로틀
            # =======================
            throttle = telemetry.get("throttle")
            if throttle is not None:
                t = np.array(throttle, dtype=float)
                result["avg_throttle"] = float(t.mean())
            else:
                result["avg_throttle"] = 0

            # =======================
            # 브레이크
            # =======================
            brake = telemetry.get("brake")
            if brake is not None:
                b = np.array(brake, dtype=float)
                result["brake_usage"] = float((b > 0).mean())
                result["max_brake"] = float(b.max())
            else:
                result["brake_usage"] = 0
                result["max_brake"] = 0

            # =======================
            # Trajectory 길이
            # =======================
            if "x" in trajectory and "y" in trajectory:
                x = np.array(trajectory["x"], dtype=float)
                y = np.array(trajectory["y"], dtype=float)
                dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                result["trajectory_length"] = float(dist.sum())
            else:
                result["trajectory_length"] = 0

        except Exception as e:
            result["error"] = f"Performance 분석 실패: {e}"

        return result
