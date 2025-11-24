import pandas as pd
from io import StringIO

class TelemetryParser:

    def parse_file(self, file_path):

        # ================================
        # 1) Raw 읽기
        # ================================
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        lines = raw.splitlines()

        # ================================
        # 2) "Time" 헤더 찾기
        # ================================
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('"Time"') or line.startswith("Time"):
                header_line = i
                break

        if header_line is None:
            raise ValueError("Telemetry header not found")

        print("[TelemetryParser] 진짜 헤더 라인 =", header_line)

        # ================================
        # 3) CSV 읽기
        # ================================
        csv_text = "\n".join(lines[header_line:])
        df = pd.read_csv(StringIO(csv_text), sep=",", engine="python")

        # ================================
        # 4) Unnamed 제거
        # ================================
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # ================================
        # 5) 컬럼명 정리
        # ================================
        df.columns = (
            df.columns.str.replace('"', "", regex=False)
                      .str.strip()
                      .str.lower()
        )

        # ================================
        # 6) Unit row 제거
        # ================================
        unit_row = df.iloc[0]

        # 첫 데이터 행이 숫자로 이루어져 있지 않으면 → unit row
        non_numeric_count = unit_row.apply(
            lambda v: pd.to_numeric(v, errors="coerce")
        ).isna().sum()

        if non_numeric_count > 3:
            print("[TelemetryParser] unit row 제거 완료")
            df = df.iloc[1:].reset_index(drop=True)

        # ================================
        # 7) 2D 컬럼 flatten (핵심)
        # ================================
        fixed = pd.DataFrame()
        for col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                print(f"[TelemetryParser] 경고: 2D 컬럼 → flatten: {col}")
                s = s.iloc[:, 0]

            fixed[col] = s

        df = fixed

        # ================================
        # 8) 모든 컬럼 숫자 변환
        # ================================
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ================================
        # 9) Outlap 제거 (distance 증가 시작점)
        # ================================
        if "distance" in df.columns:
            dist = df["distance"].fillna(0).values

            start_idx = None
            for i in range(1, len(dist)):
                if dist[i] > dist[i - 1] + 0.5:
                    start_idx = i
                    break

            if start_idx:
                print(f"[TelemetryParser] outlap 제거 → {start_idx} 행부터 시작")
                df = df.iloc[start_idx:].reset_index(drop=True)

        print("[TelemetryParser] 최종 shape =", df.shape)
        return df
