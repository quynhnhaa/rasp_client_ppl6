import os
import time
import queue
from threading import Thread
from queue import Queue

# Đặt biến môi trường cho Ultralytics để tránh cảnh báo về quyền ghi.
# Thư mục này sẽ được tạo trong cùng thư mục với script.
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_CONFIG_DIR'] = os.path.join(project_dir, '.config')

from ultralytics import YOLO
from picamera2 import Picamera2
import imagezmq
import cv2



def load_class_names(filename="class_to_id.txt"):
    """Tải danh sách tên class từ file text, mỗi class một dòng."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Đã tải {len(class_names)} class từ '{filename}'.")
        return class_names
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy file class '{filename}'. Sử dụng class mặc định ['product'].")
        return ["product"]

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    # Trỏ trực tiếp đến thư mục NCNN model
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_0284_ncnn_model"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (640, 640),
    "queue_size": 1,
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
    # class_names sẽ được load từ model, nhưng có thể giữ lại để tham khảo
    # "class_names": load_class_names("class_to_id.txt"),
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"

# ---------------------
# Thread 1: Camera
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    while True:
        frame = picam2.capture_array()
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass


# ---------------------
# Thread 2: Inference
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, result_queue: Queue):
    start_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        frame = frame_queue.get()

        # 1. Inference với ultralytics
        # Thư viện tự động xử lý pre-processing, inference và post-processing (NMS)
        results = model.predict(
            source=frame,
            conf=CONFIG["conf_threshold"],
            iou=CONFIG["nms_threshold"],
            verbose=False  # Tắt log chi tiết cho mỗi lần predict
        )

        # 2. Lấy frame đã được vẽ bounding box
        annotated_frame = results[0].plot()


        # 3. Tính toán và vẽ FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        try:
            # Đưa frame đã vẽ vào queue
            result_queue.put(annotated_frame, timeout=0.1)
        except queue.Full:
            pass


# ---------------------
# Thread 3: Gửi qua imagezmq
# ---------------------
def sender_worker(result_queue: Queue, server_address: str, camera_name: str = "raspi_cam"):
    print(f"[INFO] Dang ket noi den server tai {server_address}...")
    sender = imagezmq.ImageSender(connect_to=server_address)
    
    first_frame_sent = False
    while True:
        frame = result_queue.get()
        sender.send_image(camera_name, frame)

        if not first_frame_sent:
            print("[INFO] Client da ket noi va gui frame dau tien toi server thanh cong!")
            first_frame_sent = True


# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO11 NCNN Inference on Raspberry Pi")
    print("="*50)
    
    # 1. Khởi tạo camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print(f"[INFO] Camera started with resolution {CONFIG['camera_resolution']}")

    # 2. Load YOLO NCNN model bằng ultralytics
    print(f"[INFO] Loading NCNN model from: {CONFIG['model_name']}...")
    # Ultralytics sẽ tự tìm file .param và .bin trong thư mục được chỉ định
    model = YOLO(CONFIG['model_name'], task='detect')
    print("[INFO] Model loaded successfully.")
    # In tên các class mà model nhận diện được
    print(f"[INFO] Model classes: {model.names}")

    # 3. Tạo queue cho pipeline
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
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
    
    print("[INFO] All threads started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        picam2.stop()
        print("[INFO] Done.")