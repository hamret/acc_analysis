import cv2
import numpy as np

VIDEO_PATH = r"/dadadad.mp4"  # 영상 경로
BBOX = (821, 483, 904, 558)             # 네가 선택한 bbox

def extract_profile():
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    if not ret:
        print("❌ 첫 프레임 로딩 실패")
        return

    x1, y1, x2, y2 = BBOX
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        print("❌ BBOX cropped image is empty")
        return

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # H, S, V 각각 히스토그램
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256]).flatten()

    print("\n==============================")
    print("🎨 차량 HSV 히스토그램 생성됨 (조향 안정화용)")
    print("H hist:", hist_h[:10], " ...")
    print("S hist:", hist_s[:10], " ...")
    print("V hist:", hist_v[:10], " ...")
    print("==============================\n")

    np.save("../car_hist_h.npy", hist_h)
    np.save("../car_hist_s.npy", hist_s)
    np.save("../car_hist_v.npy", hist_v)

    print("📁 파일 저장: car_hist_h.npy, car_hist_s.npy, car_hist_v.npy")

    # 시각 확인
    cv2.imshow("crop", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

extract_profile()
