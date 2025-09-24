from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import sys

if len(sys.argv) < 2:
    print("Su dunng: python yolov11.py <model.pt>")
    sys.exit(1)

model = YOLO(sys.argv[1])


picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 640)})
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # bỏ kênh Alpha

    results = model.predict(source=frame, show=False,save=False, verbose=False)
    annoted_frame = results[0].plot()
    cv2.imshow("YOLOv11 Real-time", annoted_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.close()
