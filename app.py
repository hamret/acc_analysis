import os
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

app = Flask(__name__, static_folder="static", template_folder="templates")

app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
app.config["OUTPUT_FOLDER"] = config.OUTPUT_DIR

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

video_processor = VideoProcessor()
telemetry_parser = TelemetryParser()
trajectory_analyzer = TrajectoryAnalyzer()
sync_calibrator = SyncCalibrator()
line_warper = LineWarpEngine()
perf_analyzer = PerformanceAnalyzer()
ai_feedback = AIFeedbackEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.json
        upload_id = payload.get("upload_id")
        if not upload_id:
            return jsonify({"success": False, "error": "upload_id가 없습니다."}), 400

        # --------------------------
        # 1) 업로드된 파일 찾기
        # --------------------------
        video_path, tel_path = None, None
        for f in os.listdir(config.UPLOAD_DIR):
            if f.startswith(upload_id):
                p = os.path.join(config.UPLOAD_DIR, f)
                if f.lower().endswith(".mp4"):
                    video_path = p
                elif f.lower().endswith(".csv"):
                    tel_path = p

        if video_path is None or tel_path is None:
            return jsonify({
                "success": False,
                "error": f"업로드 ID {upload_id} 에 해당하는 mp4/csv 파일을 찾을 수 없습니다."
            }), 400

        # --------------------------
        # 2) 영상 메타 + YOLO 궤적
        # --------------------------
        meta, yolo_traj = video_processor.process(video_path)
        car_pos = yolo_traj.get("car_pos", [])

        if len(car_pos) == 0:
            return jsonify({
                "success": False,
                "error": "YOLO 트래킹 결과(car_pos)가 비어 있습니다."
            }), 500

        # --------------------------
        # 3) 텔레메트리 파싱 & Trajectory 생성
        # --------------------------
        telemetry = telemetry_parser.parse_file(tel_path)
        trajectory = trajectory_analyzer.create_trajectory(telemetry)

        # --------------------------
        # 4) Ideal line 매핑 (ACC ideal CSV)
        # --------------------------
        trajectory = trajectory_analyzer.attach_ideal_line(
            trajectory,
            "ideal_line/spa_ideal.csv"  # extract_ideal_line에서 생성
        )

        # --------------------------
        # 5) YOLO speed vs Telemetry speed 동기화
        # --------------------------
        yolo_speed = sync_calibrator.compute_yolo_speed(car_pos)
        tel_speed = telemetry["speed"].values

        offset = sync_calibrator.auto_sync_speed(yolo_speed, tel_speed)

        frame_map = sync_calibrator.generate_frame_map(
            n_video=len(yolo_speed),
            n_tel=len(tel_speed),
            offset=offset
        )
        trajectory["frame_map"] = frame_map

        # --------------------------
        # 6) 화면 좌표로 warp (real + ideal)
        # --------------------------
        warped_real, warped_ideal = line_warper.warp(
            trajectory,
            meta,
            car_pos
        )

        # --------------------------
        # 7) 최종 오버레이 영상 렌더링
        # --------------------------
        output_name = f"{upload_id}_overlay.mp4"
        output_path = os.path.join(config.OUTPUT_DIR, output_name)

        video_processor.render_overlay(
            video_path,
            warped_real,
            warped_ideal,
            yolo_traj,
            output_path
        )

        return jsonify({
            "success": True,
            "output_video": output_name
        })

    except Exception as e:
        # 디버그용 로그
        print("[ERROR] /api/analyze:", repr(e))
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
