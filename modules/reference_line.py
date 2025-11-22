import numpy as np
from scipy.interpolate import splprep, splev


class SpaReferenceLine:
    """
    Spa 서킷 기본 레이아웃(상대 좌표 기반)을 spline 보정하여
    최적 레이싱 라인 형태를 생성한다.
    """

    def __init__(self):
        pass

    def load_base_shape(self):
        """
        Spa 트랙 기본 좌표 (대략적인 형태)
        """

        pts = np.array([
            [0, 0],     # La Source
            [20, 5],
            [50, 20],   # Eau Rouge
            [60, 35],   # Raidillon
            [70, 60],
            [65, 90],   # Kemmel Straight
            [55, 120],
            [40, 150],  # Les Combes
            [20, 175],
            [0, 200],
        ], dtype=np.float32)

        return pts

    def refine_spline(self, base_pts):
        """
        기초 좌표 -> spline smoothing -> 레퍼런스 라인 생성
        """

        tck, u = splprep([base_pts[:, 0], base_pts[:, 1]], s=5.0)
        u_fine = np.linspace(0, 1, 800)

        x, y = splev(u_fine, tck)
        return np.vstack([x, y]).T  # Nx2

    def get_reference_line(self):
        base = self.load_base_shape()
        refined = self.refine_spline(base)
        return refined  # [(x,y), (x,y), ...]
