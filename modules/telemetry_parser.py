import pandas as pd
from io import StringIO


class TelemetryParser:

    def parse_file(self, file_path):

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # 1) 실제 헤더 찾기: 반드시 "Time" (따옴표 포함) 로 시작함
        header_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('"Time"'):
                header_line = i
                break

        if header_line is None:
            raise ValueError("Telemetry header not found")

        print("[TelemetryParser] 헤더 라인 =", header_line)

        # 2) 헤더 아래 전체 텍스트
        csv_text = "".join(lines[header_line:])

        # 3) 쉼표 기반 CSV (모든 값이 따옴표로 감싸져 있음)
        df = pd.read_csv(
            StringIO(csv_text),
            sep=",",
            engine="python",
            low_memory=False
        )

        # 4) Unnamed 컬럼 제거
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # 5) 숫자 컬럼 변환 시도
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        return df
