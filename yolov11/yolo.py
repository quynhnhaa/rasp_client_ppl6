from ultralytics import YOLO
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python yolov11.py <model_path>")
    sys.exit(1)

model = YOLO(sys.argv[1])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False,save=False, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 Real-time - Mac Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
