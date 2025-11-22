class AIFeedbackEngine:

    def generate_feedback(self, telemetry, trajectory, perf):

        # 존재하지 않는 key는 기본값을 넣어서 KeyError 방지
        avg_throttle = perf.get("avg_throttle", None)
        avg_speed = perf.get("avg_speed", None)
        brake_usage = perf.get("brake_usage", None)
        max_speed = perf.get("max_speed", None)

        feedback = []

        if avg_speed is not None:
            if avg_speed < 120:
                feedback.append("전체적으로 속도가 낮습니다. 코너 탈출 가속을 더 적극적으로 활용해보세요.")
            else:
                feedback.append("평균 속도가 우수합니다.")

        if avg_throttle is not None:
            if avg_throttle < 40:
                feedback.append("스로틀 사용량이 낮습니다. 직선 구간에서 가속을 더 활용해야 합니다.")
            else:
                feedback.append("스로틀 사용이 안정적으로 유지되고 있습니다.")

        if brake_usage is not None:
            if brake_usage > 0.15:
                feedback.append("브레이크 사용 빈도가 높습니다. 더 부드러운 코너 진입을 시도하세요.")
            else:
                feedback.append("브레이크 사용이 효율적입니다.")

        if max_speed is not None:
            feedback.append(f"최고 속도는 약 {max_speed:.1f} km/h 입니다.")

        # 기본 메시지
        if not feedback:
            feedback.append("주행 데이터가 충분하지 않아 상세 분석이 제한되었습니다.")

        return feedback
