# from ultralytics import YOLO
# import cv2
# from picamera2 import Picamera2
# import sys

# if len(sys.argv) < 2:
#     print("Su dunng: python yolov11.py <model.pt>")
#     sys.exit(1)

# model = YOLO(sys.argv[1])

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

# # Load the exported NCNN model
# ncnn_model = YOLO("./yolo11n_ncnn_model")

# picam2 = Picamera2()
# config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 640)})
# picam2.configure(config)
# picam2.start()

# while True:
#     frame = picam2.capture_array()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # bỏ kênh Alpha

#     results = model.predict(source=frame, show=False,save=False, verbose=False)
#     annoted_frame = results[0].plot()
#     cv2.imshow("YOLOv11 Real-time", annoted_frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cv2.destroyAllWindows()
# picam2.close()


from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import sys
import os
from time import time
if len(sys.argv) < 2:
    print("Su dung: python yolov11.py <model.pt>")
    sys.exit(1)

model_path = sys.argv[1]
model = YOLO(model_path)


export_dir = os.path.splitext(model_path)[0] + "_ncnn_model"


if not os.path.exists(export_dir):
    print(f"[INFO] Exporting {model_path} -> NCNN ...")
    model.export(format="ncnn")  
else:
    print(f"[INFO] Found existing NCNN model at: {export_dir}")

# Load NCNN model
ncnn_model = YOLO(export_dir)

# Khởi tạo camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 640)})
picam2.configure(config)
picam2.start()

while True:
    start = time()
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # bỏ kênh Alpha

    results = ncnn_model.predict(source=frame, show=False, save=False, verbose=False)
    print('FPS:', round(1/(time()-start), 3))
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 Real-time", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.close()
