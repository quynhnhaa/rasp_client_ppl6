import os
import csv
import gc
import time
import json
import queue
from threading import Thread
from multiprocessing import Process, Queue as MPQueue, Event
import yaml
from collections import Counter
import numpy as np
import paho.mqtt.client as mqtt

try:
    from RPLCD.i2c import CharLCD
    RPLCD_AVAILABLE = True
except ImportError:
    RPLCD_AVAILABLE = False
    print("[WARNING] RPLCD library not found.")

# ---------------------
# ENV
# ---------------------
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_CONFIG_DIR'] = os.path.join(project_dir, '.config')

# NOTE: Import YOLO trong process con, không import ở đây
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
        return metadata.get('imgsz', (320, 320))
    except Exception as e:
        print("[ERROR] Metadata:", e)
        return (320, 320)

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07_ncnn_model"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "queue_size": 2,
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
}

metadata_path = os.path.join(project_dir, CONFIG["model_name"], "metadata.yaml")
CONFIG["camera_resolution"] = load_class_names_from_yaml(metadata_path)
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"
del metadata_path

# ---------------------
# MQTT & LCD CONFIG
# ---------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pbl6/products"
MQTT_TOPIC_CMD = f"cmd/{CONFIG['camera_name']}"
LCD_DISPLAY_DURATION = 1.5

# ---------------------
# LCD FUNCTIONS
# ---------------------
def display_on_lcd(lcd, label, price, quantity):
    if lcd is None:
        return 0
    try:
        lcd.cursor_pos = (1, 0)
        total_price = price * quantity if quantity > 1 else price
        price_str = f"{total_price:,.0f}VND"
        lcd.write_string(price_str.rjust(16)[:16])

        quantity_str = f" x{quantity}" if quantity > 1 else ""
        quantity_len = len(quantity_str)
        label_cols = 16 - quantity_len

        lcd.cursor_pos = (0, label_cols)
        lcd.write_string(quantity_str)

        text = str(label)
        lcd.cursor_pos = (0, 0)
        if len(text) > label_cols:
            lcd.write_string(text[:label_cols])
        else:
            lcd.write_string(text.ljust(label_cols))
    except Exception as e:
        print(f"[ERROR] LCD: {e}")
    return 0

def lcd_worker(q, lcd_obj):
    if lcd_obj:
        lcd_obj.clear()
    while True:
        item = q.get()
        if item == (None, None, None):
            break
        label, price, quantity = item
        display_on_lcd(lcd_obj, label, price, quantity)
        time.sleep(LCD_DISPLAY_DURATION)
        if lcd_obj:
            lcd_obj.clear()
    print("[INFO] LCD worker stopped.")

def init_lcd():
    if not RPLCD_AVAILABLE:
        return None
    try:
        lcd = CharLCD('PCF8574', 0x27)
        lcd.clear()
        lcd.write_string("Waiting for...")
        print(f"[INFO] LCD initialized.")
        return lcd
    except Exception:
        print("[ERROR] Could not initialize LCD.")
        return None

# ---------------------
# MQTT CALLBACKS
# ---------------------
def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected. Subscribing to {MQTT_TOPIC} and {MQTT_TOPIC_CMD}")
        client.subscribe([(MQTT_TOPIC, 0), (MQTT_TOPIC_CMD, 0)])
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

def on_mqtt_message(client, userdata, msg):
    # Lấy Event từ userdata
    scanning_event = userdata.get('scanning_event')
    lcd_queue = userdata.get('lcd_queue')

    if msg.topic == MQTT_TOPIC_CMD:
        payload = msg.payload.decode().strip().upper()
        if payload == "SCAN":
            if scanning_event:
                scanning_event.set()
            print("[CMD] SCAN STARTED")
        elif payload == "STOP":
            if scanning_event:
                scanning_event.clear()
            print("[CMD] SCAN STOPPED")
        return

    # Handle product messages for LCD
    try:
        data = json.loads(msg.payload.decode())
        label = data.get("label", "N/A")
        price = data.get("price", 0)
        quantity = data.get("quantity", 1)
        if lcd_queue:
            try:
                lcd_queue.put_nowait((label, price, quantity))
            except queue.Full:
                try:
                    lcd_queue.get_nowait()
                except queue.Empty:
                    pass
                lcd_queue.put_nowait((label, price, quantity))
    except Exception as e:
        print(f"[ERROR] MQTT Message: {e}")

# ---------------------
# THREAD: CAMERA (vẫn dùng Thread vì Picamera2 không pickle được)
# ---------------------
def camera_worker(picam2, frame_queue: MPQueue, stop_event: Event):
    print("[CAMERA] Worker started.")
    while not stop_event.is_set():
        try:
            frame = picam2.capture_array()
            try:
                frame_queue.put(frame, timeout=0.05)
            except:
                pass  # Queue full, drop frame
            del frame
        except Exception as e:
            print(f"[ERROR] Camera: {e}")
            time.sleep(0.1)
    print("[CAMERA] Worker stopped.")

# ---------------------
# PROCESS: INFERENCE (dùng Process để bypass GIL)
# ---------------------
def inference_process(
    model_name: str,
    model_config: dict,
    frame_queue: MPQueue,
    result_queue: MPQueue,
    scanning_event: Event,
    stop_event: Event
):
    """
    Chạy trong process riêng biệt.
    Load model trong process này để tránh vấn đề pickle.
    """
    # Import YOLO trong process con
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from collections import Counter

    print("[INFERENCE] Process started. Loading model...")

    # Load model
    model = YOLO(model_name, task="detect")

    # Warmup
    dummy = np.zeros(
        (model_config["resolution"][1], model_config["resolution"][0], 3),
        dtype=np.uint8
    )
    model.predict(
        source=dummy,
        conf=model_config["conf_threshold"],
        iou=model_config["nms_threshold"],
        verbose=False
    )
    del dummy
    gc.collect()
    print("[INFERENCE] Model warmup complete.")

    prev_time = time.time()
    gc_counter = 0

    while not stop_event.is_set():
        # ===== 1. Lấy frame từ queue =====
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue

        # ===== 2. Tính FPS =====
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # ===== 3. Khởi tạo biến =====
        frame_counter = Counter()
        annotated = None
        is_scanning = scanning_event.is_set()

        # ===== 4. Inference hoặc hiển thị STOPPED =====
        try:
            if is_scanning:
                results = model.predict(
                    source=frame,
                    conf=model_config["conf_threshold"],
                    iou=model_config["nms_threshold"],
                    verbose=False
                )
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for cid in cls_ids:
                        frame_counter[result.names[cid]] += 1
                    del cls_ids

                annotated = result.plot(font_size=0.4, line_width=1)
                del result
                del results
            else:
                annotated = frame.copy()
                cv2.putText(
                    annotated, "STOPPED",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )
        except Exception as e:
            print(f"[ERROR] Inference: {e}")
            annotated = frame.copy() if frame is not None else None
            frame_counter = Counter()

        # ===== 5. Xóa frame gốc =====
        del frame

        # ===== 6. Vẽ FPS =====
        if annotated is not None:
            cv2.putText(
                annotated, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1
            )

        # ===== 7. Put vào result queue =====
        if annotated is not None:
            # Chuyển Counter thành dict để serialize qua multiprocessing Queue
            result_data = (annotated, dict(frame_counter), curr_time)
            try:
                result_queue.put(result_data, timeout=0.05)
            except:
                # Queue full → drop old, put new
                try:
                    old = result_queue.get_nowait()
                    del old
                except:
                    pass
                try:
                    result_queue.put_nowait(result_data)
                except:
                    pass

        # ===== 8. Cleanup =====
        del annotated
        del frame_counter

        # ===== 9. GC định kỳ =====
        gc_counter += 1
        if gc_counter >= 30:
            gc.collect()
            gc_counter = 0

    print("[INFERENCE] Process stopped.")

# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO NCNN + MULTIPROCESSING PIPELINE")
    print("=" * 50)

    # ===== Events cho đồng bộ =====
    scanning_event = Event()  # Thay thế IS_SCANNING global
    stop_event = Event()      # Signal để dừng các worker

    # ===== Camera =====
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()
    print("[INFO] Camera started.")

    # ===== LCD =====
    lcd_queue = queue.Queue(maxsize=3)  # Thread-safe queue cho LCD
    lcd = init_lcd()
    Thread(target=lcd_worker, args=(lcd_queue, lcd), daemon=True).start()

    # ===== MQTT =====
    mqtt_client = mqtt.Client()
    mqtt_client.user_data_set({
        'lcd_queue': lcd_queue,
        'scanning_event': scanning_event
    })
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[ERROR] MQTT Connection: {e}")

    # ===== Multiprocessing Queues =====
    frame_queue = MPQueue(maxsize=CONFIG["queue_size"])
    result_queue = MPQueue(maxsize=CONFIG["queue_size"])

    # ===== Config cho inference process =====
    model_config = {
        "resolution": CONFIG["camera_resolution"],
        "conf_threshold": CONFIG["conf_threshold"],
        "nms_threshold": CONFIG["nms_threshold"],
    }

    # ===== Start Camera Thread =====
    camera_thread = Thread(
        target=camera_worker,
        args=(picam2, frame_queue, stop_event),
        daemon=True
    )
    camera_thread.start()

    # ===== Start Inference Process =====
    inference_proc = Process(
        target=inference_process,
        args=(
            CONFIG["model_name"],
            model_config,
            frame_queue,
            result_queue,
            scanning_event,
            stop_event
        )
    )
    inference_proc.start()
    print(f"[INFO] Inference process started (PID: {inference_proc.pid})")

    # ===== ImageZMQ Sender (Main Thread) =====
    sender = imagezmq.ImageSender(connect_to=CONFIG["server_address"])
    print(f"[INFO] Connected to server {CONFIG['server_address']}")

    print("[INFO] System running. Ctrl+C to stop.")

    try:
        while True:
            try:
                frame, counter, timestamp = result_queue.get(timeout=1.0)
                msg = {
                    "camera_name": CONFIG["camera_name"],
                    "counter": counter,  # Đã là dict
                    "time": timestamp
                }
                sender.send_image(json.dumps(msg), frame)
                del frame
                del counter
                del timestamp
            except:
                continue

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")

    finally:
        # ===== Cleanup =====
        stop_event.set()

        # Đợi inference process kết thúc
        inference_proc.join(timeout=3)
        if inference_proc.is_alive():
            print("[WARNING] Force terminating inference process...")
            inference_proc.terminate()

        # Dừng MQTT
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

        # Dừng LCD
        if lcd:
            lcd_queue.put((None, None, None))
            lcd.close(clear=True)

        # Dừng camera
        picam2.stop()

        # Dọn queues
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                break
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                break

        print("[INFO] Stopped.")