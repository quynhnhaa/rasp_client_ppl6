from ultralytics import YOLO
import cv2
import sys
import os
from time import time

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

cap = cv2.VideoCapture(0)

while True:
    start = time()
    ret, frame = cap.read()
    if not ret:
        break

    results = ncnn_model.predict(source=frame, show=False,save=False, verbose=False)
    print('FPS:', round(1/(time()-start), 3))
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 Real-time - Mac Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
