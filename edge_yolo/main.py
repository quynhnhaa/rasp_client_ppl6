import os
import json
import csv
import gc
import time
import queue
from threading import Thread
from queue import Queue
import yaml
from collections import Counter
from enum import Enum

# ---------------------
# ENV
# ---------------------
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_CONFIG_DIR'] = os.path.join(project_dir, '.config')

from ultralytics import YOLO
from picamera2 import Picamera2
import imagezmq
import cv2

# ---------------------
# LOAD METADATA
# ---------------------
def load_class_names_from_yaml(metadata_path):
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        class_names = [name for _, name in sorted(metadata['names'].items())]
        return class_names, metadata.get('imgsz', (320, 320))
    except Exception as e:
        print("[ERROR] Metadata:", e)
        return ["product"], (320, 320)

def load_price_map(file_path):
    price_map = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get('label', '').strip()
                price_str = row.get('price', '0').strip()
                if label and price_str:
                    try:
                        price_map[label] = int(price_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[ERROR] Loading prices: {e}")
    return price_map

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07_ncnn_model"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "queue_size": 2, # Tăng buffer lên 3 để tránh blocking dây chuyền
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
}

metadata_path = os.path.join(project_dir, CONFIG["model_name"], "metadata.yaml")
CONFIG["class_names"], CONFIG["camera_resolution"] = load_class_names_from_yaml(metadata_path)
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"

price_map_path = os.path.join(project_dir, "product_price.csv")
PRICE_MAP = load_price_map(price_map_path)

# ---------------------
# FSM
# ---------------------
class ScanState(Enum):
    IDLE = 0
    SCANNING = 1

EMPTY_TIMEOUT = 1.2  # giây

# ---------------------
# THREAD 1: CAMERA
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    while True:
        frame = picam2.capture_array()
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass

# ---------------------
# THREAD 2: INFERENCE ONLY
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, detection_queue: Queue, sender_frame_queue: Queue):
    prev_time = time.time()
    count_gc = 0
    while True:
        frame = frame_queue.get()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        results = model.predict(
            source=frame,
            conf=CONFIG["conf_threshold"],
            iou=CONFIG["nms_threshold"],
            verbose=False
        )

        result = results[0]
        frame_counter = Counter()

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for cid in cls_ids:
                frame_counter[result.names[cid]] += 1

        annotated = result.plot(font_size=0.4, line_width=1)

        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Đẩy frame trực tiếp cho Sender
        sender_frame_queue.put(annotated)

        # Chỉ đẩy metadata cho Scan FSM
        detection_queue.put({
            "time": time.time(),
            "counter": frame_counter
        })

        # --- GIẢI PHÓNG BỘ NHỚ THỦ CÔNG ---
        # Xóa các biến nặng ngay lập tức để tránh OOM trên Pi Zero 2W
        del frame
        del results
        del result
        # annotated đã được put vào queue, sender sẽ lo, ở đây ta xóa tham chiếu cục bộ
        del annotated
        
        # Ép chạy Garbage Collection mỗi 30 frame để dọn sạch RAM
        count_gc += 1
        if count_gc > 30:
            gc.collect()
            count_gc = 0

# ---------------------
# THREAD 3: SCAN FSM
# ---------------------
def scan_fsm_worker(detection_queue: Queue, result_queue: Queue):
    state = ScanState.IDLE
    last_seen_time = 0
    batch_frame_counters = []
    session_total = Counter()  # Mục "đã quét" (tổng các đợt)
    current_scanning = {}    # Mục "đang quét" (đợt hiện tại)
    while True:
        data = detection_queue.get()
        frame_counter = data["counter"]
        now = data["time"]  

        if state == ScanState.IDLE:
            if frame_counter:
                state = ScanState.SCANNING
                batch_frame_counters = [frame_counter]
                last_seen_time = now
                print("\n[SCAN] START")

        elif state == ScanState.SCANNING:
            if frame_counter:
                batch_frame_counters.append(frame_counter)
                last_seen_time = now
            
            # Cập nhật mục "đang quét" từ dữ liệu batch hiện có
            if batch_frame_counters:
                votes = Counter(tuple(sorted(c.items())) for c in batch_frame_counters)
                if votes:
                    best, _ = votes.most_common(1)[0]
                    current_scanning = dict(best)

            if not frame_counter:
                if now - last_seen_time > EMPTY_TIMEOUT:
                    # Điều kiện thoả mãn: Kết thúc đợt -> Chuyển từ "đang quét" sang "đã quét"
                    if current_scanning:
                        if len(batch_frame_counters) < 5:
                            print(f"[SCAN] IGNORED (Too few frames: {len(batch_frame_counters)})")
                        else:
                            print("[SCAN] END →", current_scanning)
                            session_total.update(current_scanning)
                            
                            # Tính tổng tiền dựa trên PRICE_MAP đã load
                            total_money = sum(PRICE_MAP.get(k, 0) * v for k, v in session_total.items())
                            print(f"[SESSION TOTAL] → {dict(session_total)} | Money: {total_money:,} VND")

                        current_scanning = {}

                    state = ScanState.IDLE
                    batch_frame_counters = []
                # else: Điều kiện sai -> Vẫn giữ ở mục "đang quét" (current_scanning)

        result_queue.put({
            "current": current_scanning,
            "total": dict(session_total)
        })

# ---------------------
# THREAD 4: SENDER
# ---------------------
def sender_worker(result_queue: Queue, sender_frame_queue: Queue, server_address: str, camera_name: str):
    sender = imagezmq.ImageSender(connect_to=server_address)
    print(f"[INFO] Connected to server {server_address}")

    while True:
        # Lấy frame và dữ liệu từ 2 nguồn khác nhau (tự động đồng bộ vì quy trình 1-1)
        frame = sender_frame_queue.get()
        data = result_queue.get()
        
        msg = {
            "camera_name": camera_name,
            "current": data["current"],
            "total": data["total"]
        }
        sender.send_image(json.dumps(msg), frame)

# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO NCNN + FSM SCAN PIPELINE")
    print("=" * 50)

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Model
    model = YOLO(CONFIG["model_name"], task="detect")

    # Queues
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    detection_queue = Queue(maxsize=CONFIG["queue_size"])
    result_queue = Queue(maxsize=CONFIG["queue_size"])
    sender_frame_queue = Queue(maxsize=CONFIG["queue_size"])

    # Threads
    Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True).start()
    Thread(target=inference_worker, args=(model, frame_queue, detection_queue, sender_frame_queue), daemon=True).start()
    Thread(target=scan_fsm_worker, args=(detection_queue, result_queue), daemon=True).start()
    Thread(target=sender_worker, args=(result_queue, sender_frame_queue, CONFIG["server_address"], CONFIG["camera_name"]), daemon=True).start()

    print("[INFO] System running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        picam2.stop()
        print("\n[INFO] Stopped.")
