import os
import csv
import gc
import time
import json
import queue
import threading
from threading import Thread, Lock, Event
from queue import Queue
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

from ultralytics import YOLO
from picamera2 import Picamera2
import imagezmq
import cv2

# ---------------------
# MEMORY MONITORING
# ---------------------
def get_memory_usage():
    """Đọc memory usage từ /proc/meminfo (không cần psutil)"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(':')
                value = int(parts[1])
                mem_info[key] = value
        
        total = mem_info.get('MemTotal', 1)
        available = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
        used = total - available
        percent = (used / total) * 100
        return {
            'total_mb': total / 1024,
            'used_mb': used / 1024,
            'available_mb': available / 1024,
            'percent': percent
        }
    except:
        return {'percent': 0, 'available_mb': 999}

def emergency_memory_cleanup():
    """Dọn dẹp khẩn cấp khi memory cao"""
    gc.collect()
    gc.collect()  # Gọi 2 lần để dọn circular references
    time.sleep(0.1)

# ---------------------
# THREAD-SAFE SCAN STATE
# ---------------------
class ScanStateManager:
    def __init__(self):
        self._is_scanning = False
        self._lock = Lock()
        self._last_change_time = 0
        self._min_interval = 0.5  # Tối thiểu 0.5s giữa các lần đổi trạng thái
    
    @property
    def is_scanning(self):
        with self._lock:
            return self._is_scanning
    
    def set_scanning(self, value: bool) -> bool:
        """Set trạng thái, return True nếu thành công"""
        current_time = time.time()
        with self._lock:
            # Rate limiting - không cho đổi quá nhanh
            if current_time - self._last_change_time < self._min_interval:
                print(f"[WARNING] State change too fast, ignored")
                return False
            
            if self._is_scanning != value:
                self._is_scanning = value
                self._last_change_time = current_time
                # Force GC khi đổi trạng thái
                gc.collect()
                return True
            return False

scan_state = ScanStateManager()

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
    "queue_size": 1,
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

# Shutdown event for graceful exit
shutdown_event = Event()

# ========== LCD Functions ==========

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
    
    while not shutdown_event.is_set():
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
            
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

# ========== MQTT Callbacks ==========

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected. Subscribing...")
        client.subscribe([(MQTT_TOPIC, 0), (MQTT_TOPIC_CMD, 0)])
    else:
        print(f"[MQTT] Failed to connect, rc={rc}")

def on_mqtt_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_CMD:
        payload = msg.payload.decode().strip().upper()
        if payload == "SCAN":
            if scan_state.set_scanning(True):
                print("[CMD] SCAN STARTED")
            else:
                print("[CMD] SCAN ignored (rate limited)")
        elif payload == "STOP":
            if scan_state.set_scanning(False):
                print("[CMD] SCAN STOPPED")
            else:
                print("[CMD] STOP ignored (rate limited)")
        return

    q = userdata.get('queue')
    try:
        data = json.loads(msg.payload.decode())
        label = data.get("label", "N/A")
        price = data.get("price", 0)
        quantity = data.get("quantity", 1)
        if q:
            try:
                q.put_nowait((label, price, quantity))
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait((label, price, quantity))
    except Exception as e:
        print(f"[ERROR] MQTT Message: {e}")

def on_mqtt_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected (rc={rc})")
    if not shutdown_event.is_set():
        print("[MQTT] Attempting reconnect...")
        while not shutdown_event.is_set():
            try:
                client.reconnect()
                print("[MQTT] Reconnected!")
                break
            except Exception:
                time.sleep(5)

# ---------------------
# THREAD: CAMERA
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    print("[INFO] Camera worker started")
    consecutive_errors = 0
    
    while not shutdown_event.is_set():
        try:
            frame = picam2.capture_array()
            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                del frame  # Quan trọng: giải phóng frame nếu queue đầy
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"[ERROR] Camera: {e}")
            if consecutive_errors > 10:
                print("[FATAL] Too many camera errors, stopping")
                shutdown_event.set()
                break
            time.sleep(0.5)
    
    print("[INFO] Camera worker stopped")

# ---------------------
# THREAD: INFERENCE (SỬA LỖI CHÍNH)
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, sender_frame_queue: Queue):
    print("[INFO] Inference worker started")
    
    prev_time = time.time()
    frame_count = 0
    last_gc_time = time.time()
    last_memory_check = time.time()
    
    # Warmup model
    dummy = np.zeros(
        (CONFIG["camera_resolution"][1], CONFIG["camera_resolution"][0], 3),
        dtype=np.uint8
    )
    model.predict(
        source=dummy,
        conf=CONFIG["conf_threshold"],
        iou=CONFIG["nms_threshold"],
        verbose=False
    )
    del dummy
    gc.collect()
    print("[INFO] Model warmup complete")
    
    # Pre-allocate reusable objects
    empty_counter = Counter()
    
    while not shutdown_event.is_set():
        # ===== MEMORY CHECK (mỗi 5 giây) =====
        current_time = time.time()
        if current_time - last_memory_check > 5:
            mem = get_memory_usage()
            if mem['percent'] > 80:
                # print(f"[WARNING] High memory: {mem['percent']:.1f}% ({mem['available_mb']:.0f}MB free)")
                emergency_memory_cleanup()
            if mem['percent'] > 90:
                print(f"[CRITICAL] Memory critical: {mem['percent']:.1f}%")
                # Tạm dừng scan để giải phóng memory
                scan_state.set_scanning(False)
                emergency_memory_cleanup()
                time.sleep(1)
            last_memory_check = current_time
        
        # ===== GET FRAME =====
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # ===== CALCULATE FPS =====
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        frame_count += 1

        # ===== PROCESS FRAME =====
        frame_counter = Counter()
        annotated = None
        
        is_scanning = scan_state.is_scanning  # Đọc 1 lần, tránh race condition
        
        if is_scanning:
            try:
                results = model.predict(
                    source=frame,
                    conf=CONFIG["conf_threshold"],
                    iou=CONFIG["nms_threshold"],
                    verbose=False
                )
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for cid in cls_ids:
                        frame_counter[result.names[cid]] += 1

                # Plot với kích thước nhỏ hơn để tiết kiệm memory
                annotated = result.plot(font_size=0.4, line_width=1)
                
                # QUAN TRỌNG: Giải phóng ngay lập tức
                del results
                del result
                if 'cls_ids' in dir():
                    del cls_ids
                    
            except Exception as e:
                print(f"[ERROR] Inference: {e}")
                annotated = frame.copy()
                cv2.putText(annotated, "ERROR", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Không scan - chỉ copy frame và vẽ text
            annotated = frame.copy()
            cv2.putText(annotated, "STOPPED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Giải phóng frame gốc ngay
        del frame

        # ===== DRAW FPS =====
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # ===== SEND TO QUEUE =====
        try:
            sender_frame_queue.put_nowait((annotated, frame_counter, curr_time))
        except queue.Full:
            try:
                old_item = sender_frame_queue.get_nowait()
                # Giải phóng item cũ
                del old_item
            except queue.Empty:
                pass
            try:
                sender_frame_queue.put_nowait((annotated, frame_counter, curr_time))
            except queue.Full:
                del annotated
                del frame_counter
                continue

        # ===== GARBAGE COLLECTION =====
        # GC thường xuyên hơn trên Pi Zero 2W
        if curr_time - last_gc_time > 2.0:  # Mỗi 2 giây
            gc.collect()
            last_gc_time = curr_time
        
        # Force GC mỗi 50 frames
        if frame_count % 50 == 0:
            gc.collect()

    print("[INFO] Inference worker stopped")

# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO NCNN + FSM SCAN PIPELINE")
    print(f"Initial Memory: {get_memory_usage()['percent']:.1f}%")
    print("=" * 50)

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print(f"[INFO] Camera started at {CONFIG['camera_resolution']}")

    # Model
    model = YOLO(CONFIG["model_name"], task="detect")
    gc.collect()
    print(f"[INFO] Model loaded. Memory: {get_memory_usage()['percent']:.1f}%")

    # LCD & MQTT Setup
    lcd_queue = Queue(maxsize=3)
    lcd = init_lcd()
    Thread(target=lcd_worker, args=(lcd_queue, lcd), daemon=True).start()

    mqtt_client = mqtt.Client()
    mqtt_client.user_data_set({'queue': lcd_queue})
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.on_disconnect = on_mqtt_disconnect
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[ERROR] MQTT Connection: {e}")

    # Queues
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    sender_frame_queue = Queue(maxsize=CONFIG["queue_size"])

    # Threads
    camera_thread = Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True)
    inference_thread = Thread(target=inference_worker, args=(model, frame_queue, sender_frame_queue), daemon=True)
    
    camera_thread.start()
    inference_thread.start()

    print("[INFO] System running. Ctrl+C to stop.")
    print(f"[INFO] Memory after threads: {get_memory_usage()['percent']:.1f}%")

    # ImageZMQ with timeout
    sender = imagezmq.ImageSender(connect_to=CONFIG["server_address"])
    print(f"[INFO] Connected to server {CONFIG['server_address']}")

    try:
        send_count = 0
        last_memory_log = time.time()
        
        while not shutdown_event.is_set():
            try:
                frame, counter, timestamp = sender_frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            msg = {
                "camera_name": CONFIG["camera_name"],
                "counter": dict(counter),
                "time": timestamp
            }
            
            try:
                sender.send_image(json.dumps(msg), frame)
            except Exception as e:
                print(f"[ERROR] Send failed: {e}")
            
            # Giải phóng ngay
            del frame
            del counter
            del msg
            
            send_count += 1
            
            # Log memory mỗi 30 giây
            if time.time() - last_memory_log > 30:
                mem = get_memory_usage()
                print(f"[STATS] Sent: {send_count}, Memory: {mem['percent']:.1f}%, Free: {mem['available_mb']:.0f}MB")
                last_memory_log = time.time()
                
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        
    finally:
        shutdown_event.set()
        time.sleep(0.5)
        
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        
        if lcd:
            lcd.clear()
            lcd.close()
            
        picam2.stop()
        
        # Final cleanup
        gc.collect()
        print(f"[INFO] Final memory: {get_memory_usage()['percent']:.1f}%")
        print("[INFO] Stopped.")