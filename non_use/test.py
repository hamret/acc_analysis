import pandas as pd

telemetry_path = r"C:\Users\user\PycharmProjects\acc_analysis\uploads\20251124_153726_Spa-porsche_992_gt3_r-2-2025.11.19-19.22.19.csv"

# 헤더 라인 찾기
with open(telemetry_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

header_line = None
for i, line in enumerate(lines):
    if "time" in line and "speed" in line and "roty" in line:
        header_line = i
        break

print("header line =", header_line)

# CSV 로드
df = pd.read_csv(telemetry_path, header=header_line)

print(df["distance"].head(20))
print(df["distance"].tail(20))
