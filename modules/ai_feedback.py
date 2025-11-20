class AIFeedbackEngine:
    def generate_feedback(self, telemetry, traj, perf):
        fb = []

        if perf["avg_throttle"] < 40:
            fb.append("→ 전체적으로 쓰로틀 사용률이 낮아 가속 여지가 있습니다.")

        if perf["avg_brake"] > 30:
            fb.append("→ 제동 사용량이 높습니다. 코너 진입 속도를 조절해보세요.")

        if perf["avg_steer"] > 20:
            fb.append("→ 조향각 변동이 크므로 레이싱 라인을 더 부드럽게 잡아보세요.")

        if not fb:
            fb.append("좋습니다! 안정적인 주행 패턴이 감지되었습니다.")

        return fb
