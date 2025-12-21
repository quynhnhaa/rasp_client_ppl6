import os
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
# MEMORY CONSTANTS - Điều chỉnh cho Pi Zero 2W
# ---------------------
MEMORY_WARNING_THRESHOLD = 70      # Cảnh báo sớm hơn
MEMORY_CRITICAL_THRESHOLD = 78     # Dừng scan sớm hơn
MEMORY_EMERGENCY_THRESHOLD = 85    # Dừng hoàn toàn
MIN_FREE_MB_FOR_INFERENCE = 100    # Cần ít nhất 100MB free để chạy inference

# ---------------------
# MEMORY MONITORING
# ---------------------
def get_memory_usage():
    """Đọc memory usage từ /proc/meminfo"""
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
        return {'percent': 0, 'available_mb': 999, 'total_mb': 512, 'used_mb': 0}

def aggressive_memory_cleanup():
    """Dọn dẹp memory mạnh mẽ"""
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Cố gắng giải phóng memory cho kernel
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('1')
    except:
        pass  # Cần quyền root, bỏ qua nếu không có
    
    time.sleep(0.2)

def can_run_inference():
    """Kiểm tra có đủ RAM để chạy inference không"""
    mem = get_memory_usage()
    return (mem['available_mb'] >= MIN_FREE_MB_FOR_INFERENCE and 
            mem['percent'] < MEMORY_CRITICAL_THRESHOLD)

# ---------------------
# THREAD-SAFE SCAN STATE
# ---------------------
class ScanStateManager:
    def __init__(self):
        self._is_scanning = False
        self._lock = Lock()
        self._last_change_time = 0
        self._min_interval = 1.0  # Tăng lên 1 giây
        self._paused_by_memory = False
    
    @property
    def is_scanning(self):
        with self._lock:
            return self._is_scanning and not self._paused_by_memory
    
    @property
    def is_paused(self):
        with self._lock:
            return self._paused_by_memory
    
    def set_scanning(self, value: bool) -> bool:
        current_time = time.time()
        with self._lock:
            if current_time - self._last_change_time < self._min_interval:
                return False
            
            if self._is_scanning != value:
                self._is_scanning = value
                self._last_change_time = current_time
                self._paused_by_memory = False
                return True
            return False
    
    def pause_for_memory(self):
        """Tạm dừng do thiếu RAM"""
        with self._lock:
            if not self._paused_by_memory:
                self._paused_by_memory = True
                print("[MEMORY] Inference paused - low memory")
    
    def resume_from_pause(self):
        """Tiếp tục sau khi có đủ RAM"""
        with self._lock:
            if self._paused_by_memory:
                self._paused_by_memory = False
                print("[MEMORY] Inference resumed")

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

shutdown_event = Event()

# ========== LCD Functions ==========

def display_on_lcd(lcd, label, price, quantity):
    if lcd is None:
        return
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
        lcd.write_string("Waiting...")
        return lcd
    except Exception:
        print("[ERROR] Could not initialize LCD.")
        return None

# ========== MQTT Callbacks ==========

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected")
        client.subscribe([(MQTT_TOPIC, 0), (MQTT_TOPIC_CMD, 0)])
    else:
        print(f"[MQTT] Failed, rc={rc}")

def on_mqtt_message(client, userdata, msg):
    if msg.topic == MQTT_TOPIC_CMD:
        payload = msg.payload.decode().strip().upper()
        if payload == "SCAN":
            # Kiểm tra memory trước khi cho phép scan
            if can_run_inference():
                if scan_state.set_scanning(True):
                    print("[CMD] SCAN STARTED")
            else:
                print("[CMD] SCAN DENIED - insufficient memory")
        elif payload == "STOP":
            if scan_state.set_scanning(False):
                print("[CMD] SCAN STOPPED")
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
        print(f"[ERROR] MQTT: {e}")

def on_mqtt_disconnect(client, userdata, rc):
    if not shutdown_event.is_set() and rc != 0:
        print(f"[MQTT] Disconnected, reconnecting...")
        while not shutdown_event.is_set():
            try:
                client.reconnect()
                break
            except:
                time.sleep(5)

# ---------------------
# THREAD: CAMERA
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    print("[INFO] Camera worker started")
    
    while not shutdown_event.is_set():
        try:
            frame = picam2.capture_array()
            try:
                frame_queue.put(frame, timeout=0.05)
            except queue.Full:
                del frame
        except Exception as e:
            print(f"[ERROR] Camera: {e}")
            time.sleep(0.5)
    
    print("[INFO] Camera worker stopped")

# ---------------------
# THREAD: INFERENCE - TỐI ƯU CHO PI ZERO 2W
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, sender_frame_queue: Queue):
    print("[INFO] Inference worker started")
    
    prev_time = time.time()
    frame_count = 0
    last_memory_check = time.time()
    consecutive_skips = 0
    
    # Pre-create stopped frame text
    stopped_text = "STOPPED"
    paused_text = "PAUSED-MEM"
    
    # Warmup với memory check
    print("[INFO] Warming up model...")
    mem_before = get_memory_usage()
    print(f"[INFO] Memory before warmup: {mem_before['percent']:.1f}%")
    
    dummy = np.zeros(
        (CONFIG["camera_resolution"][1], CONFIG["camera_resolution"][0], 3),
        dtype=np.uint8
    )
    model.predict(source=dummy, conf=CONFIG["conf_threshold"], 
                  iou=CONFIG["nms_threshold"], verbose=False)
    del dummy
    
    # Force cleanup sau warmup
    aggressive_memory_cleanup()
    
    mem_after = get_memory_usage()
    print(f"[INFO] Memory after warmup: {mem_after['percent']:.1f}%")
    print("[INFO] Model warmup complete")
    
    while not shutdown_event.is_set():
        current_time = time.time()
        
        # ===== MEMORY CHECK - Mỗi 2 giây =====
        if current_time - last_memory_check > 2.0:
            mem = get_memory_usage()
            last_memory_check = current_time
            
            if mem['percent'] > MEMORY_EMERGENCY_THRESHOLD:
                print(f"[EMERGENCY] Memory {mem['percent']:.1f}% - forcing cleanup")
                scan_state.pause_for_memory()
                aggressive_memory_cleanup()
                time.sleep(0.5)
                continue
                
            elif mem['percent'] > MEMORY_CRITICAL_THRESHOLD:
                if not scan_state.is_paused:
                    print(f"[CRITICAL] Memory {mem['percent']:.1f}% - pausing inference")
                    scan_state.pause_for_memory()
                    gc.collect()
                    
            elif mem['percent'] < MEMORY_WARNING_THRESHOLD:
                if scan_state.is_paused:
                    scan_state.resume_from_pause()
        
        # ===== GET FRAME =====
        try:
            frame = frame_queue.get(timeout=0.5)
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
        
        is_scanning = scan_state.is_scanning
        is_paused = scan_state.is_paused
        
        if is_scanning and not is_paused:
            # Double-check memory before inference
            mem = get_memory_usage()
            if mem['available_mb'] < MIN_FREE_MB_FOR_INFERENCE:
                # Không đủ RAM - skip inference frame này
                consecutive_skips += 1
                annotated = frame  # Dùng frame gốc, không copy
                cv2.putText(annotated, f"LOW MEM ({mem['available_mb']:.0f}MB)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                if consecutive_skips > 5:
                    scan_state.pause_for_memory()
                    gc.collect()
            else:
                consecutive_skips = 0
                try:
                    # Run inference
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
                        del cls_ids

                    annotated = result.plot(font_size=0.4, line_width=1)
                    
                    # Cleanup ngay lập tức
                    del result
                    del results
                    
                except Exception as e:
                    print(f"[ERROR] Inference: {e}")
                    annotated = frame
                    cv2.putText(annotated, "ERROR", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        elif is_paused:
            annotated = frame  # Không copy để tiết kiệm RAM
            cv2.putText(annotated, paused_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            annotated = frame
            cv2.putText(annotated, stopped_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ===== DRAW FPS & MEMORY =====
        mem = get_memory_usage()
        cv2.putText(annotated, f"FPS:{fps:.1f} MEM:{mem['percent']:.0f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # ===== SEND TO QUEUE =====
        try:
            sender_frame_queue.put_nowait((annotated, frame_counter, curr_time))
        except queue.Full:
            try:
                old = sender_frame_queue.get_nowait()
                del old
            except queue.Empty:
                pass
            try:
                sender_frame_queue.put_nowait((annotated, frame_counter, curr_time))
            except:
                pass

        # Không del frame ở đây vì annotated có thể là reference đến frame
        
        # ===== GC mỗi 30 frames =====
        if frame_count % 30 == 0:
            gc.collect()

    print("[INFO] Inference worker stopped")

# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO NCNN - OPTIMIZED FOR PI ZERO 2W")
    print("=" * 50)
    
    initial_mem = get_memory_usage()
    print(f"[INIT] Total RAM: {initial_mem['total_mb']:.0f}MB")
    print(f"[INIT] Used: {initial_mem['used_mb']:.0f}MB ({initial_mem['percent']:.1f}%)")
    print(f"[INIT] Available: {initial_mem['available_mb']:.0f}MB")

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print(f"[INFO] Camera: {CONFIG['camera_resolution']}")
    gc.collect()

    # Model
    print("[INFO] Loading model...")
    model = YOLO(CONFIG["model_name"], task="detect")
    gc.collect()
    
    mem_after_model = get_memory_usage()
    print(f"[INFO] Memory after model: {mem_after_model['percent']:.1f}%")

    # LCD & MQTT
    lcd_queue = Queue(maxsize=2)
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
        print(f"[ERROR] MQTT: {e}")

    # Queues - size nhỏ để tiết kiệm RAM
    frame_queue = Queue(maxsize=1)
    sender_frame_queue = Queue(maxsize=1)

    # Threads
    Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True).start()
    Thread(target=inference_worker, args=(model, frame_queue, sender_frame_queue), daemon=True).start()

    print("[INFO] System running. Ctrl+C to stop.")

    # ImageZMQ
    sender = imagezmq.ImageSender(connect_to=CONFIG["server_address"])
    print(f"[INFO] Server: {CONFIG['server_address']}")

    try:
        send_count = 0
        last_log = time.time()
        
        while not shutdown_event.is_set():
            try:
                data = sender_frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            frame, counter, timestamp = data
            del data
                
            msg = {
                "camera_name": CONFIG["camera_name"],
                "counter": dict(counter),
                "time": timestamp
            }
            
            try:
                sender.send_image(json.dumps(msg), frame)
            except Exception as e:
                print(f"[ERROR] Send: {e}")
            
            del frame, counter, msg
            send_count += 1
            
            # Log mỗi 60 giây
            if time.time() - last_log > 60:
                mem = get_memory_usage()
                print(f"[STATS] Sent:{send_count} Mem:{mem['percent']:.1f}% Free:{mem['available_mb']:.0f}MB")
                last_log = time.time()
                gc.collect()
                
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        
    finally:
        shutdown_event.set()
        time.sleep(0.5)
        mqtt_client.loop_stop()
        if lcd:
            lcd.clear()
            lcd.close()
        picam2.stop()
        print("[INFO] Stopped.")