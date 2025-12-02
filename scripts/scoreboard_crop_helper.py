import cv2

VIDEO_PATH = r"C:\Users\v5karthi\Desktop\OntarioOpen_Mat5_Sat2025.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        return

    # let user select ROI
    roi = cv2.selectROI("Select Scoreboard Region", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    print("\nSelected ROI (x1, y1, x2, y2):")
    x, y, w, h = roi
    print(f"({x}, {y}, {x + w}, {y + h})")

if __name__ == "__main__":
    main()

