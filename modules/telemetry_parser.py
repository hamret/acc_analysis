import pandas as pd
import re
from io import StringIO


class TelemetryParser:
    """
    ACC MoTeC CSV 파서
    """

    REQUIRED = {
        "time": "Time",
        "distance": "Distance",
        "speed": "SPEED",
        "throttle": "THROTTLE",
        "brake": "BRAKE",
        "steerangle": "STEERANGLE",
        "rpms": "RPMS",
    }

    def parse_file(self, path):
        print("[TelemetryParser] CSV 자동 분석 시작...")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        # 빈 줄 제거
        clean = "\n".join([line for line in raw.split("\n") if line.strip()])

        # 헤더 찾기
        header_idx = None
        lines = clean.split("\n")

        for i, line in enumerate(lines):
            if line.startswith("Time,"):
                header_idx = i
                break

        if header_idx is None:
            raise RuntimeError("데이터 시작 라인을 찾을 수 없음")

        print(f"[TelemetryParser] 헤더 라인: {header_idx}")

        # 데이터 영역만 추출
        data_text = "\n".join(lines[header_idx:])

        df = pd.read_csv(StringIO(data_text))

        # 컬럼 강제 소문자화
        df.columns = [c.lower() for c in df.columns]

        # 누락 컬럼 보정
        for key, col in self.REQUIRED.items():
            c = col.lower()
            if c not in df.columns:
                print(f"[TelemetryParser] 누락된 컬럼 자동 생성: {c}")
                df[c] = 0.0

        df = df.reset_index(drop=True)
        print("[TelemetryParser] CSV 파싱 완료!")

        return df
