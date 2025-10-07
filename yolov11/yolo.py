# from ultralytics import YOLO
# import cv2
# import sys
# import os
# from time import time

# model_path = sys.argv[1]
# model = YOLO(model_path)


# export_dir = os.path.splitext(model_path)[0] + "_ncnn_model"


# if not os.path.exists(export_dir):
#     print(f"[INFO] Exporting {model_path} -> NCNN ...")
#     model.export(format="ncnn")  
# else:
#     print(f"[INFO] Found existing NCNN model at: {export_dir}")

# # Load NCNN model
# model = YOLO(export_dir)

# cap = cv2.VideoCapture(0)

# while True:
#     start = time()
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.predict(source=frame, show=False,save=False, verbose=False)
#     print('FPS:', round(1/(time()-start), 3))
#     annotated_frame = results[0].plot()
#     cv2.imshow("YOLOv11 Real-time - Mac Camera", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import sys
import os
from time import time, sleep
from multiprocessing import Process, Queue

def camera_process(frame_queue):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            sleep(0.01) 
    cap.release()

def inference_process(frame_queue, model_path):
    model = YOLO(model_path)

    # # Kiểm tra NCNN export
    # export_dir = os.path.splitext(model_path)[0] + "_ncnn_model"
    # if not os.path.exists(export_dir):
    #     print(f"[INFO] Exporting {model_path} -> NCNN ...")
    #     model.export(format="ncnn")
    # else:
    #     print(f"[INFO] Found existing NCNN model at: {export_dir}")

    # # Load NCNN model
    # model = YOLO(export_dir)


    frame_index = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            start = time()
            results = model.predict(frame,device="mps", show=False, save=False, verbose=False)
            fps = round(1/(time()-start), 2)
            if frame_index % 10 == 0:
                print("FPS (inference):", fps)
            frame_index += 1

            annotated = results[0].plot()
            cv2.imshow("YOLOv11 Real-time - Mac Camera", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Su dung: python yolov11_mp.py <model.pt>")
        sys.exit(1)

    model_path = sys.argv[1]
    frame_queue = Queue(maxsize=5)

    p1 = Process(target=camera_process, args=(frame_queue,))
    p2 = Process(target=inference_process, args=(frame_queue, model_path))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
