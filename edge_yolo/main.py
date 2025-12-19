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
import paho.mqtt.client as mqtt

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
        del frame

# ---------------------
# THREAD 2: INFERENCE ONLY
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, detection_queue: Queue, sender_frame_queue: Queue):
    prev_time = time.time()
    count_gc = 0
    while True:
        # frame = frame_queue.get()
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue


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

        # SỬA: Dùng put_nowait + try-except để không block
        try:
            sender_frame_queue.put_nowait(annotated)
        except queue.Full:
            # Drop frame cũ, put frame mới
            try:
                sender_frame_queue.get_nowait()
            except queue.Empty:
                pass
            sender_frame_queue.put_nowait(annotated)

        try:
            detection_queue.put_nowait({
                "time": time.time(),
                "counter": frame_counter
            })
        except queue.Full:
            try:
                detection_queue.get_nowait()
            except queue.Empty:
                pass
            detection_queue.put_nowait({
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
        del frame_counter
        
        # Ép chạy Garbage Collection mỗi 30 frame để dọn sạch RAM
        count_gc += 1
        if count_gc > 30:
            gc.collect()
            count_gc = 0

# ---------------------
# THREAD 3: SCAN FSM
# ---------------------
def scan_fsm_worker(detection_queue: Queue, mqtt_queue: Queue):
    state = ScanState.IDLE
    last_seen_time = 0
    batch_frame_counters = []
    session_total = Counter()  # Mục "đã quét" (tổng các đợt)
    current_scanning = {}    # Mục "đang quét" (đợt hiện tại)
    
    last_mqtt_send_time = 0
    MQTT_INTERVAL = 0.5  # Giới hạn gửi MQTT mỗi 0.5s (tránh spam)

    while True:
        try:
            data = detection_queue.get(timeout=1.0)
        except queue.Empty:
            continue

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
                    del best
                del votes
            if not frame_counter:
                if now - last_seen_time > EMPTY_TIMEOUT:
                    # Điều kiện thoả mãn: Kết thúc đợt -> Chuyển từ "đang quét" sang "đã quét"
                    if current_scanning:
                        if len(batch_frame_counters) < 5:
                            # print(f"[SCAN] IGNORED (Too few frames: {len(batch_frame_counters)})")
                            pass
                        else:
                            print("[SCAN] END →", current_scanning)
                            session_total.update(current_scanning)

                        current_scanning.clear()

                    state = ScanState.IDLE
                    batch_frame_counters.clear()
                # else: Điều kiện sai -> Vẫn giữ ở mục "đang quét" (current_scanning)

        # Gửi dữ liệu qua MQTT (có giới hạn thời gian)
        if time.time() - last_mqtt_send_time > MQTT_INTERVAL:
            try:
                mqtt_queue.put_nowait({
                    "current": current_scanning,
                    "total": dict(session_total)
                })
                last_mqtt_send_time = time.time()
            except queue.Full:
                pass

        del data
        del frame_counter

# ---------------------
# THREAD 5: MQTT WORKER
# ---------------------
def mqtt_worker(mqtt_queue: Queue, mqtt_client, camera_name: str):
    while True:
        data = mqtt_queue.get()
        payload = json.dumps(data)
        mqtt_client.publish(f"scan/{camera_name}", payload)
        del data
        del payload

# ---------------------
# THREAD 4: SENDER
# ---------------------
def sender_worker(sender_frame_queue: Queue, server_address: str, camera_name: str):
    try:
        sender.send_image(camera_name, frame)
    except Exception as e:
        print("[SENDER ERROR]", e)
        sender = imagezmq.ImageSender(connect_to=server_address)

    print(f"[INFO] Connected to server {server_address}")

    while True:
        # Chỉ lấy frame từ sender_frame_queue và gửi đi ngay lập tức
        frame = sender_frame_queue.get()
        sender.send_image(camera_name, frame)
        del frame

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

    # MQTT
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(CONFIG["server_ip"], 1883, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[ERROR] MQTT Connection: {e}")

    # Queues
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    detection_queue = Queue(maxsize=CONFIG["queue_size"])
    sender_frame_queue = Queue(maxsize=CONFIG["queue_size"])
    mqtt_queue = Queue(maxsize=CONFIG["queue_size"])

    # Threads
    Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True).start()
    Thread(target=inference_worker, args=(model, frame_queue, detection_queue, sender_frame_queue), daemon=True).start()
    Thread(target=scan_fsm_worker, args=(detection_queue, mqtt_queue), daemon=True).start()
    Thread(target=mqtt_worker, args=(mqtt_queue, mqtt_client, CONFIG["camera_name"]), daemon=True).start()
    Thread(target=sender_worker, args=(sender_frame_queue, CONFIG["server_address"], CONFIG["camera_name"]), daemon=True).start()

    print("[INFO] System running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        picam2.stop()
        print("\n[INFO] Stopped.")