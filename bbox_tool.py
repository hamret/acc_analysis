import cv2

VIDEO_PATH = r"C:\Users\user\PycharmProjects\acc_analysis\dadadad.mp4"   # 네가 업로드한 영상 경로 그대로 사용

drawing = False
ix, iy = -1, -1
bbox = None


def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, bbox

    # 마우스 왼쪽 버튼 누름 → 시작점 기록
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = None

    # 드래그 중 → 박스 형태 표시
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            bbox = (ix, iy, x, y)

    # 마우스 왼쪽 버튼 떼면 → bbox 완료
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x, y)
        print(f"📦 BBOX = ({ix}, {iy}, {x}, {y})")


def main():
    global bbox

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 로드 실패:", VIDEO_PATH)
        return

    print("▶ 첫 프레임 로딩 중...")
    ret, frame = cap.read()
    if not ret:
        print("❌ 첫 프레임 읽기 실패")
        return

    cv2.namedWindow("bbox_tool")
    cv2.setMouseCallback("bbox_tool", draw_bbox)

    print("📌 마우스로 드래그해서 박스를 만드세요.")
    print("💡 완료 후 ENTER 키를 누르면 bbox가 확정됩니다.")

    while True:
        temp = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("bbox_tool", temp)
        key = cv2.waitKey(10)

        if key == 13:  # ENTER
            break
        if key == 27:  # ESC
            bbox = None
            break

    cv2.destroyAllWindows()

    if bbox:
        x1, y1, x2, y2 = bbox
        print("\n==============================")
        print("🎯 최종 선택된 BBOX 좌표")
        print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print("==============================\n")
    else:
        print("❌ bbox 선택 안됨")


if __name__ == "__main__":
    main()
