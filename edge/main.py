import os
import time
import queue
from threading import Thread
from queue import Queue

from picamera2 import Picamera2
import imagezmq
import cv2
import ncnn
import numpy as np
import yaml


def load_class_names_from_yaml(metadata_path):
    """Tải danh sách tên class từ file metadata.yaml của model."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        # `names` là một dictionary {id: name}, chúng ta cần lấy list các name
        # Sắp xếp theo key (id) để đảm bảo thứ tự đúng
        class_names = [name for _, name in sorted(metadata['names'].items())]
        print(f"[INFO] Đã tải {len(class_names)} class từ '{metadata_path}'.")
        return class_names
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file metadata: {e}. Sử dụng class mặc định.")
        return ["product"] # Trả về giá trị mặc định nếu có lỗi

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07.pt"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (320, 320),
    "queue_size": 1,
    "input_size": (320, 320),  # Phải khớp với imgsz khi export
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
    "input_layer": "in0",
    "output_layer": "out0",
    "num_threads": 4,  # Số threads cho inference
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Hàm load NCNN model
# ---------------------
def load_ncnn_model(model_name):
    # if not os.path.isfile(model_name):
    #     print(f"[ERROR] Khong tim thay file mo hinh: {model_name}")
    #     raise FileNotFoundError(model_name)
        
    export_dir = os.path.splitext(model_name)[0] + "_ncnn_model"
    param_path = os.path.join(export_dir, "model.ncnn.param")
    bin_path = os.path.join(export_dir, "model.ncnn.bin")
    metadata_path = os.path.join(export_dir, "metadata.yaml")
    CONFIG["class_names"] = load_class_names_from_yaml(metadata_path)

    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(f"Khong tim thay file .param hoac .bin trong: {export_dir}")
    CONFIG["input_layer"] = "in0"
    CONFIG["output_layer"] = "out0"

    net = ncnn.Net()
    
    # ĐẶT SỐ THREADS Ở ĐÂY - trên net.opt
    net.opt.use_vulkan_compute = False  # Tắt Vulkan nếu không có GPU
    net.opt.num_threads = CONFIG["num_threads"]  # Số threads cho inference
    
    net.load_param(param_path)
    net.load_model(bin_path)
    print(f"[INFO] Da load NCNN model tu: {export_dir}")
    print(f"[INFO] Num threads: {CONFIG['num_threads']}")
    return net


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
# Postprocess cho YOLO11
# Dựa trên logic từ file yolo.py
# ---------------------
def postprocess(frame, outputs, conf_threshold, nms_threshold, class_names):
    """Hậu xử lý output của mô hình để lấy bounding box."""
    frame_height, frame_width = frame.shape[:2]
    input_height, input_width = CONFIG["input_size"]

    # Tỷ lệ scale giữa ảnh gốc và ảnh input
    x_factor = frame_width / input_width
    y_factor = frame_height / input_height

    boxes = []
    scores = []
    class_ids = []

    # Output của NCNN có thể là (1, 5, 2100) hoặc (5, 2100)
    # Bỏ chiều batch nếu có và chuyển vị (transpose) để có shape (num_boxes, 4_coords + num_classes)
    detections = np.squeeze(outputs).T

    for row in detections:
        # Lấy điểm tin cậy của các class (từ cột thứ 4 trở đi)
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]

        if max_score > conf_threshold:
            # Lấy tọa độ bounding box (cx, cy, w, h)
            cx, cy, w, h = row[:4]

            # Chuyển đổi tọa độ về kích thước ảnh gốc
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(int(class_id))

    # Áp dụng Non-Maximum Suppression để loại bỏ các box trùng lặp
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return frame

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        
        # Kiểm tra an toàn để tránh lỗi IndexError
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"ID_{class_id}: {score:.2f}" # Hiển thị ID nếu không tìm thấy tên class

        # Vẽ bounding box và label lên ảnh
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


# ---------------------
# Thread 2: Inference
# ---------------------
def inference_worker(net: ncnn.Net, frame_queue: Queue, result_queue: Queue):
    start_time = time.time()
    frame_count = 0
    fps = 0
    first_inference = True

    input_w, input_h = CONFIG["input_size"]

    while True:
        frame = frame_queue.get()

        # 1. Pre-processing
        mat_in = ncnn.Mat.from_pixels_resize(
            frame, 
            ncnn.Mat.PixelType.PIXEL_RGB,
            frame.shape[1], 
            frame.shape[0], 
            input_w, 
            input_h
        )
        
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])
        # mat_in.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        # 2. Inference - KHÔNG cần set_num_threads ở đây
        ex = net.create_extractor()
        
        ret = ex.input(CONFIG["input_layer"], mat_in)
        if ret != 0:
            print(f"[ERROR] Failed to set input. Return code: {ret}")
            continue
            
        ret, out = ex.extract(CONFIG["output_layer"])
        if ret != 0:
            print(f"[ERROR] Failed to extract output. Return code: {ret}")
            continue
            
        outputs = np.array(out)

        # Debug on first inference
        if first_inference:
            print(f"[DEBUG] First inference output shape: {outputs.shape}")
            print(f"[DEBUG] Output dtype: {outputs.dtype}")
            print(f"[DEBUG] Output min/max: {outputs.min():.4f}/{outputs.max():.4f}")
            first_inference = False

        # 3. Post-processing
        annotated_frame = postprocess(
            frame, 
            outputs, 
            CONFIG["conf_threshold"], 
            CONFIG["nms_threshold"],
            CONFIG["class_names"]
        )

        # Calculate and draw FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    print("="*50)
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

    # 2. Load YOLO NCNN model
    model = load_ncnn_model(CONFIG["model_name"])
    print(f"[INFO] Number of classes: {len(CONFIG['class_names'])}")

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