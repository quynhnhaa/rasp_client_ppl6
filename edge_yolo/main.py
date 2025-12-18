import os
import time
import queue
from threading import Thread
from queue import Queue
import yaml

# Đặt biến môi trường cho Ultralytics để tránh cảnh báo về quyền ghi.
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_CONFIG_DIR'] = os.path.join(project_dir, '.config')

from ultralytics import YOLO
from picamera2 import Picamera2
import imagezmq
import cv2



def load_class_names_from_yaml(metadata_path):
    """Tải danh sách tên class từ file metadata.yaml của model."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        # `names` là một dictionary {id: name}, chúng ta cần lấy list các name
        # Sắp xếp theo key (id) để đảm bảo thứ tự đúng
        class_names = [name for _, name in sorted(metadata['names'].items())]
        print(f"[INFO] Đã tải {len(class_names)} class từ '{metadata_path}'.")
        return class_names, metadata.get('imgsz', (320, 320))
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy file metadata '{metadata_path}'.")
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file metadata: {e}")
    return ["product"], (320, 320) # Trả về giá trị mặc định nếu có lỗi

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07_ncnn_model"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "queue_size": 1,
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
}

# Đọc class names và resolution từ metadata.yaml
metadata_path = os.path.join(project_dir, CONFIG["model_name"], "metadata.yaml")
class_names, resolution = load_class_names_from_yaml(metadata_path)
CONFIG["class_names"] = class_names
CONFIG["camera_resolution"] = tuple(resolution) if isinstance(resolution, list) else resolution

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
            verbose=False 
        )

        # 2. Lấy frame đã được vẽ bounding box
        # Có thể tùy chỉnh kích thước chữ của label bằng tham số font_size
        annotated_frame = results[0].plot(
            font_size=0.4,  # Giảm kích thước font của label
            line_width=1    # Đồng thời giảm độ dày của bounding box cho phù hợp
        )

        # 3. Tính toán và vẽ FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        try:
            result_queue.put(annotated_frame, timeout=0.1)
        except queue.Full:
            pass


# ---------------------
# Thread 3: Gửi qua imagezmq
# ---------------------
def sender_worker(result_queue: Queue, server_address: str, camera_name: str = "raspi_cam"):
    print(f"[INFO] Dang ket noi den server tai {server_address}...")
    sender = imagezmq.ImageSender(connect_to=server_address)
    
    while True:
        frame = result_queue.get()
        sender.send_image(camera_name, frame)



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
    model = YOLO(CONFIG['model_name'], task='detect')
    print("[INFO] Model loaded successfully.")
    # In tên các class mà model nhận diện được
    print(f"[INFO] Model classes (từ file metadata): {CONFIG['class_names']}")

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