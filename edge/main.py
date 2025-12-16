import os
import time
import queue
from threading import Thread
from queue import Queue

from picamera2 import Picamera2
from ultralytics import YOLO
import imagezmq
import cv2  # dùng nếu cần chuyển BGR/RGB, vẽ box, etc.

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_0284.pt"),
    "server_ip": os.getenv("SERVER_IP", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (640, 640),
    "queue_size": 5,
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Hàm load NCNN model
# ---------------------
def load_ncnn_model():
    model_name = CONFIG["model_name"]
    
    if not os.path.isfile(model_name):
        print(f"[ERROR] Khong tim thay file mo hinh: {model_name}")
        raise FileNotFoundError(model_name)

    export_dir = os.path.splitext(model_name)[0] + "_ncnn_model"

    if not os.path.exists(export_dir):
        print(f"[INFO] Chua tim thay {export_dir}. Bat dau export NCNN...")
        model = YOLO(model_name)
        model.export(format="ncnn")
        print(f"[DONE] Export NCNN thanh cong tai: {export_dir}")
    else:
        print(f"[INFO] Da ton tai NCNN model tai: {export_dir}")

    print(f"[INFO] Loading NCNN model tu thu muc: {export_dir}")
    ncnn_model = YOLO(export_dir)
    print("[DONE] Load NCNN model thanh cong")
    return ncnn_model

# ---------------------
# Thread 1: Camera
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    while True:
        frame = picam2.capture_array()   # frame là numpy array (RGB/BGR tùy config)
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            # Nếu queue đầy thì bỏ frame này (tránh lag do dồn frame cũ)
            pass

# ---------------------
# Thread 2: Inference
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, result_queue: Queue):
    # FPS calculation variables
    start_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        frame = frame_queue.get()  # block đến khi có frame

        # Tùy camera config mà frame có thể là RGB/BGR, chỉnh lại nếu cần
        # results = model(frame, imgsz=640)[0]
        results = model(frame, verbose=False)[0]   # đơn giản, verbose=False để tắt log của YOLO

        # Vẽ bounding box lên frame (tuỳ bạn, có thể gửi raw + bbox riêng)
        annotated_frame = results.plot()  # trả về numpy array BGR

        # Calculate and draw FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        try:
            result_queue.put(annotated_frame, timeout=0.1)
        except queue.Full:
            pass

# ---------------------
# Thread 3: Gửi qua imagezmq
# ---------------------
def sender_worker(result_queue: Queue, server_address: str, camera_name: str = "raspi_cam"):
    sender = imagezmq.ImageSender(connect_to=server_address)
    while True:
        frame = result_queue.get()  # block đến khi có dữ liệu
        # frame ở đây là numpy array (BGR), imagezmq hỗ trợ trực tiếp
        sender.send_image(camera_name, frame)

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    # 1. Khởi tạo camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": CONFIG["camera_resolution"]})
    picam2.configure(config)
    picam2.start()

    # 2. Load YOLO NCNN model
    model = load_ncnn_model()

    # 3. Tạo queue cho pipeline
    frame_queue = Queue(maxsize=CONFIG["queue_size"])   # giới hạn để tránh tràn RAM
    result_queue = Queue(maxsize=CONFIG["queue_size"])

    # 4. Khởi chạy 3 thread
    cam_thread = Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True)
    inf_thread = Thread(target=inference_worker, args=(model, frame_queue, result_queue), daemon=True)
    send_thread = Thread(
        target=sender_worker,
        args=(result_queue, CONFIG["server_address"], CONFIG["camera_name"]),
        daemon=True
    )

    cam_thread.start()
    inf_thread.start()
    send_thread.start()

    # 5. Giữ main thread sống, có thể thêm xử lý signal/thoát
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        # Có thể thêm code dừng camera, đóng socket, etc.
        picam2.stop()