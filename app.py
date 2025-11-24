import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory

import config
from modules.video_processor import VideoProcessor
from modules.telemetry_parser import TelemetryParser
from modules.trajectory_analyzer import TrajectoryAnalyzer
from modules.sync_calibrator import SyncCalibrator
from modules.line_warp import LineWarpEngine
from modules.performance_analyzer import PerformanceAnalyzer
from modules.ai_feedback import AIFeedbackEngine


# ===============================================================
# Flask App
# ===============================================================
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
app.config["OUTPUT_FOLDER"] = config.OUTPUT_DIR

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# ===============================================================
# Engines
# ===============================================================
video_processor = VideoProcessor()
telemetry_parser = TelemetryParser()
trajectory_analyzer = TrajectoryAnalyzer()
sync_calibrator = SyncCalibrator()
line_warper = LineWarpEngine()
perf_analyzer = PerformanceAnalyzer()
ai_feedback = AIFeedbackEngine()


# ===============================================================
# Pages
# ===============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html")


# ===============================================================
# Upload API
# ===============================================================
@app.route("/api/upload", methods=["POST"])
def upload_files():
    video_file = request.files.get("video")
    tel_file = request.files.get("telemetry")

    if not video_file or not tel_file:
        return jsonify({"success": False, "error": "영상 또는 텔레메트리 파일이 누락되었습니다."})

    upload_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_name = f"{upload_id}_{video_file.filename}"
    tel_name = f"{upload_id}_{tel_file.filename}"

    video_path = os.path.join(config.UPLOAD_DIR, video_name)
    tel_path = os.path.join(config.UPLOAD_DIR, tel_name)

    video_file.save(video_path)
    tel_file.save(tel_path)

    return jsonify({
        "success": True,
        "upload_id": upload_id,
        "video_path": video_name,
        "csv_path": tel_name
    })


# ===============================================================
# Main Analysis API
# ===============================================================
@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.json
    upload_id = payload.get("upload_id")

    if not upload_id:
        return jsonify({"success": False, "error": "upload_id가 전달되지 않았습니다."})

    # 파일 검색
    video_path, tel_path = None, None
    for f in os.listdir(config.UPLOAD_DIR):
        if f.startswith(upload_id):
            lower = f.lower()
            full = os.path.join(config.UPLOAD_DIR, f)
            if lower.endswith(".mp4"):
                video_path = full
            elif lower.endswith(".csv"):
                tel_path = full

    if not video_path:
        return jsonify({"success": False, "error": "업로드된 영상 파일을 찾을 수 없습니다."})
    if not tel_path:
        return jsonify({"success": False, "error": "업로드된 텔레메트리 CSV를 찾을 수 없습니다."})

    print(f"[ANALYZE] 업로드 ID = {upload_id}")
    print(f"영상 = {video_path}")
    print(f"CSV = {tel_path}")

    try:
        # 1) 영상 분석
        meta, yolo_traj = video_processor.process(video_path)

        # 2) CSV 텔레메트리 파싱
        telemetry = telemetry_parser.parse_file(tel_path)

        # 3) XY 주행 라인 생성
        trajectory = trajectory_analyzer.create_trajectory(telemetry)

        # ================================
        # 3) YOLO distance progression
        # ================================
        yolo_dist = trajectory_analyzer.create_yolo_distance(yolo_traj["car_pos"])

        # ================================
        # 4) 텔레메트리 distance (원본)
        # ================================
        tel_dist = telemetry["distance"].values

        # ================================
        # 5) distance 기반 frame_map 생성
        # ================================
        frame_map = sync_calibrator.generate_frame_map_by_distance(yolo_dist, tel_dist)

        trajectory["frame_map"] = frame_map

        # 5) 레이싱 라인 Warp
        warped_lines = line_warper.warp_lines_to_video_view(
            trajectory,
            meta,
            yolo_traj["car_pos"]
        )

        # 6) 오버레이 영상 렌더링
        output_name = f"{upload_id}_overlay.mp4"
        output_path = os.path.join(config.OUTPUT_DIR, output_name)

        video_processor.render_overlay(
            video_path, warped_lines, yolo_traj, output_path
        )

        # 7) 성능 분석
        performance = perf_analyzer.analyze(telemetry, trajectory)

        # 8) AI 피드백 생성 (🔥 인자 3개 넣어줘야 함)
        feedback = ai_feedback.generate_feedback(telemetry, trajectory, performance)

        return jsonify({
            "success": True,
            "output_video": output_name,
            "performance": performance,
            "feedback": feedback
        })

    except Exception as e:
        print("\n======== 내부 오류 발생 ========")
        import traceback
        traceback.print_exc()
        print("================================\n")

        return jsonify({
            "success": False,
            "error": str(e)
        })


# ===============================================================
# Outputs
# ===============================================================
@app.route("/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


# ===============================================================
# Run Server
# ===============================================================
if __name__ == "__main__":
    print("======================================================")
    print(" ACC Driving Analyzer Server Started")
    print("======================================================")
    print(f"UPLOAD_DIR = {config.UPLOAD_DIR}")
    print(f"OUTPUT_DIR = {config.OUTPUT_DIR}")
    print("======================================================")

    app.run(host="0.0.0.0", port=5000, debug=True)
