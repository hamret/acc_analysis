import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev


class IdealLineExtractor:

    def extract(self, map_path="static/maps/Spa-Map.png"):
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("트랙 맵 이미지를 읽을 수 없습니다.")

        # 1) Edge detect
        edges = cv2.Canny(img, 80, 160)

        # 2) 가장 긴 외곽 컨투어를 트랙으로 가정
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        track = max(contours, key=len)
        pts = track[:, 0, :]  # (N, 2)

        # 3) Spline 보간으로 부드러운 선 생성 (주행 라인 근사)
        x = pts[:, 0]
        y = pts[:, 1]

        tck, u = splprep([x, y], s=800.0, per=True)
        unew = np.linspace(0, 1, 5000)
        out = splev(unew, tck)

        cx = np.array(out[0])
        cy = np.array(out[1])

        # 4) 누적 거리 / distance_norm 계산
        dist = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        dist = np.insert(dist, 0, 0)
        dist = np.cumsum(dist)

        df = pd.DataFrame({
            "pixel_x": cx,
            "pixel_y": cy,
            "distance_raw": dist,
        })
        df["distance_norm"] = df["distance_raw"] / df["distance_raw"].max()

        os.makedirs("ideal_line", exist_ok=True)
        out_path = os.path.join("ideal_line", "spa_ideal.csv")
        df.to_csv(out_path, index=False)
        print(f"[IdealLine] {out_path} 생성 완료")
