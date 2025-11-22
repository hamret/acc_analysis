import numpy as np


class TrajectoryAnalyzer:

    def create_trajectory(self, telemetry):
        """
        텔레메트리에서 XY 궤적을 생성하는 기본 분석기
        → 실제 스파 프랑코샴 좌표계를 쓰지 않고,
          distance + steer 기반의 상대적인 패스 생성
        """

        print("[ANALYZE] 텔레메트리 XY 변환 시작...")

        # 필수 컬럼 체크
        required = ["distance", "steerangle"]
        for col in required:
            if col not in telemetry.columns:
                raise RuntimeError(f"'{col}' 컬럼이 텔레메트리에 존재하지 않습니다.")

        # 숫자로 강제 변환 (문자열 혼입 방지)
        dist = telemetry["distance"].astype(float).to_numpy()
        steer = telemetry["steerangle"].astype(float).to_numpy()

        # NaN 제거
        dist = np.nan_to_num(dist)
        steer = np.nan_to_num(steer)

        # 스케일링
        scale = 0.05   # distance → x 좌표 변환
        steer_gain = 0.005  # 조향 → y 편향

        # x(t)
        x = dist * scale

        # y(t) = 누적 조향량 기반 궤적
        # steering 값은 보통 -20 ~ 20deg → 이걸 살짝 누적해서 좌우 궤적 생성
        y = np.cumsum(steer * steer_gain)

        # 정상적으로 float 배열인지 확인
        x = x.astype(float)
        y = y.astype(float)

        trajectory = {
            "x": x.tolist(),
            "y": y.tolist(),
            "distance": dist.tolist(),
            "steer": steer.tolist()
        }

        print("[ANALYZE] 텔레메트리 XY 변환 완료!")
        return trajectory
