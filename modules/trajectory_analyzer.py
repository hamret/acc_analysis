import numpy as np
import pandas as pd


class TrajectoryAnalyzer:

    def create_trajectory(self, telemetry):
        time = telemetry["time"].to_numpy()
        yaw_rate_deg = telemetry["roty"].to_numpy()
        speed_kmh = telemetry["speed"].to_numpy()
        dist_raw = telemetry["distance"].to_numpy()

        # heading 적분
        dt = np.diff(time, prepend=time[0])
        dt = np.clip(dt, 0.001, 0.2)

        yaw_rate_rad = np.radians(yaw_rate_deg)
        heading = np.cumsum(yaw_rate_rad * dt)

        # distance + heading → XY
        x = dist_raw * np.cos(heading)
        y = dist_raw * np.sin(heading)

        traj = {
            "x": x.tolist(),
            "y": y.tolist(),
            "heading": heading.tolist(),
            "speed": speed_kmh.tolist(),
            "distance": dist_raw.tolist(),
        }
        return traj

    def attach_ideal_line(self, trajectory, ideal_path):
        """
        ideal_line/spa_ideal.csv :
        pixel_x, pixel_y, distance_raw, distance_norm ...
        """
        ideal = pd.read_csv(ideal_path)

        tel_d = np.array(trajectory["distance"])
        tel_norm = (tel_d - tel_d.min()) / (tel_d.max() - tel_d.min() + 1e-9)

        ideal_norm = ideal["distance_norm"].values

        # 각 텔레 포인트마다 가장 가까운 ideal distance_norm 인덱스 찾기
        mapping = [int(np.argmin(np.abs(ideal_norm - d))) for d in tel_norm]

        trajectory["ideal_x"] = ideal["pixel_x"].values[mapping]
        trajectory["ideal_y"] = ideal["pixel_y"].values[mapping]

        return trajectory
