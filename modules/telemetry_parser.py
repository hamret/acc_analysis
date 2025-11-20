import pandas as pd
import re
from io import StringIO

class TelemetryParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        print("[TelemetryParser] CSV 자동 분석 시작...")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        # 1) 따옴표 제거 + 공백 정리
        clean = raw.replace('""', '"')

        # 2) 줄 단위 분리
        lines = clean.splitlines()

        # 3) 실제 헤더 라인 찾기
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"Time"') or line.startswith('Time,'):
                header_idx = i
                break

        if header_idx is None:
            raise RuntimeError("데이터 시작 라인을 찾을 수 없음")

        print(f"[TelemetryParser] 헤더 라인 = {header_idx}")

        # 4) 단위 라인 = header_idx + 1
        unit_idx = header_idx + 1

        header_line = lines[header_idx]
        unit_line = lines[unit_idx]

        # 5) CSV 본문만 추출
        data_text = "\n".join(lines[header_idx:])

        # 6) pandas로 읽기
        df = pd.read_csv(StringIO(data_text))

        print("[TelemetryParser] 원본 컬럼:")
        print(list(df.columns))

        # 7) 컬럼명 정규화 (소문자 + 공백 제거)
        df.columns = [c.strip().lower() for c in df.columns]

        # 8) 실제 필요한 컬럼 rename
        rename_map = {
            "steerangle": "steer",
            "speed": "speed",
            "time": "time",
            "distance": "distance",
            "throttle": "throttle",
            "brake": "brake",
            "rpms": "rpms",
        }

        df = df.rename(columns=rename_map)

        # 9) 누락 컬럼 자동 생성
        essential_cols = ["time", "distance", "speed", "throttle", "brake", "steer", "rpms"]

        for col in essential_cols:
            if col not in df:
                print(f"[TelemetryParser] 누락된 컬럼 자동 생성: {col}")
                df[col] = 0.0

        print("[TelemetryParser] CSV 파싱 완료!")
        return df
