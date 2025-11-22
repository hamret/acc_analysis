import pandas as pd
from io import StringIO


class TelemetryParser:

    def parse_file(self, file_path):

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # 1) 헤더 라인 찾기
        header_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('"Time"'):
                header_line = i
                break

        if header_line is None:
            raise ValueError("Telemetry header not found")

        print("[TelemetryParser] 헤더 라인 =", header_line)

        # 2) CSV 내용만 추출
        csv_text = "".join(lines[header_line:])

        # 3) CSV 읽기
        df = pd.read_csv(
            StringIO(csv_text),
            sep=",",
            low_memory=False
        )

        # 4) unnamed 제거
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # 5) 컬럼 소문자화
        df.columns = [c.lower() for c in df.columns]

        # 6) Unit row 제거 (distance="m", time="s" 등)
        unit_row = df.iloc[0]
        if not pd.to_numeric(unit_row, errors="coerce").notna().all():
            df = df.iloc[1:].reset_index(drop=True)

        # 7) 숫자 변환
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df
