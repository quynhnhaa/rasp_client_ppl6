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

# ---------------------
# Hàm helper để tải class names
# ---------------------
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


def get_ncnn_layer_names(param_path):
    """Đọc file .param để lấy tên input và output layer."""
    input_name = None
    output_name = None
    
    with open(param_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            layer_type = parts[0]
            layer_name = parts[1]
            
            # Input layer
            if layer_type == "Input":
                input_name = layer_name
            
            # Tìm output layer (thường là layer cuối hoặc có tên chứa 'output')
            if "output" in layer_name.lower() or layer_type in ["Concat", "Permute", "Reshape"]:
                output_name = layer_name
    
    # Fallback values cho Ultralytics export
    if input_name is None:
        input_name = "in0"
    if output_name is None:
        output_name = "out0"
    
    print(f"[INFO] Input layer: {input_name}, Output layer: {output_name}")
    return input_name, output_name


# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_0284.pt"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (640, 640),
    "queue_size": 1,
    "input_size": (640, 640),  # Phải khớp với imgsz khi export
    "conf_threshold": 0.25,
    "nms_threshold": 0.45,
    "class_names": load_class_names("class_to_id.txt"),
    "input_layer": "in0",   # Sẽ được cập nhật sau
    "output_layer": "out0", # Sẽ được cập nhật sau
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Hàm load NCNN model
# ---------------------
def load_ncnn_model(model_name):
    if not os.path.isfile(model_name):
        print(f"[ERROR] Khong tim thay file mo hinh: {model_name}")
        raise FileNotFoundError(model_name)
        
    export_dir = os.path.splitext(model_name)[0] + "_ncnn_model"
    param_path = os.path.join(export_dir, "model.ncnn.param")
    bin_path = os.path.join(export_dir, "model.ncnn.bin")

    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(f"Khong tim thay file .param hoac .bin trong: {export_dir}")

    # Lấy tên layer từ file param
    input_name, output_name = get_ncnn_layer_names(param_path)
    CONFIG["input_layer"] = input_name
    CONFIG["output_layer"] = output_name

    net = ncnn.Net()
    # Tối ưu cho Raspberry Pi
    net.opt.use_vulkan_compute = False  # Tắt Vulkan nếu không có GPU
    net.opt.num_threads = 4  # Số threads phù hợp với Pi
    
    net.load_param(param_path)
    net.load_model(bin_path)
    print(f"[INFO] Da load NCNN model tu: {export_dir}")
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
# ---------------------
def postprocess_yolo11(frame, outputs, conf_threshold, nms_threshold, num_classes):
    """
    Post-process YOLO11 NCNN output.
    
    YOLO11/YOLOv8 output format từ Ultralytics NCNN export:
    - Shape: [4 + num_classes, num_boxes] 
    - Ví dụ: [84, 8400] cho 80 classes với input 640x640
    - Row 0-3: x_center, y_center, width, height (pixel values)
    - Row 4+: class probabilities
    """
    h, w = frame.shape[:2]
    input_w, input_h = CONFIG["input_size"]
    
    # Scale factors
    x_factor = w / input_w
    y_factor = h / input_h

    boxes = []
    scores = []
    class_ids = []

    # Debug shape
    print(f"[DEBUG] Output shape: {outputs.shape}")

    # Handle different output shapes
    # NCNN output có thể là [C, N] hoặc [1, C, N] hoặc [N, C]
    if len(outputs.shape) == 3:
        outputs = outputs.squeeze(0)
    
    # Kiểm tra shape và transpose nếu cần
    # YOLO11 output: [4 + num_classes, num_boxes]
    # Ta cần transpose thành [num_boxes, 4 + num_classes]
    if outputs.shape[0] == (4 + num_classes):
        outputs = outputs.T
    elif outputs.shape[1] == (4 + num_classes):
        pass  # Đã đúng format
    else:
        print(f"[WARNING] Unexpected output shape: {outputs.shape}")
        # Thử transpose nếu dimension đầu nhỏ hơn
        if outputs.shape[0] < outputs.shape[1]:
            outputs = outputs.T

    print(f"[DEBUG] Output shape after transpose: {outputs.shape}")

    for detection in outputs:
        # detection: [x_center, y_center, width, height, class_0, class_1, ...]
        class_scores = detection[4:4 + num_classes]
        class_id = int(np.argmax(class_scores))
        max_score = float(class_scores[class_id])

        if max_score > conf_threshold:
            cx, cy, box_w, box_h = detection[:4]

            # Convert center format to corner format
            # Scale to original image size
            x1 = int((cx - box_w / 2) * x_factor)
            y1 = int((cy - box_h / 2) * y_factor)
            box_width = int(box_w * x_factor)
            box_height = int(box_h * y_factor)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            box_width = min(box_width, w - x1)
            box_height = min(box_height, h - y1)

            boxes.append([x1, y1, box_width, box_height])
            scores.append(max_score)
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    final_boxes = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                final_boxes.append({
                    "box": (x, y, x + bw, y + bh),
                    "score": scores[i],
                    "class_id": class_ids[i]
                })
    
    return final_boxes


# ---------------------
# Thread 2: Inference
# ---------------------
def inference_worker(net: ncnn.Net, frame_queue: Queue, result_queue: Queue):
    start_time = time.time()
    frame_count = 0
    fps = 0
    first_inference = True

    input_w, input_h = CONFIG["input_size"]
    num_classes = len(CONFIG["class_names"])

    while True:
        frame = frame_queue.get()

        # 1. Pre-processing
        # YOLO expects RGB normalized to [0, 1]
        mat_in = ncnn.Mat.from_pixels_resize(
            frame, 
            ncnn.Mat.PixelType.PIXEL_RGB,  # Camera đã xuất RGB888
            frame.shape[1], 
            frame.shape[0], 
            input_w, 
            input_h
        )
        
        # Normalize: (pixel - mean) * norm = pixel * (1/255)
        # mean = [0, 0, 0], norm = [1/255, 1/255, 1/255]
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

        # 2. Inference
        ex = net.create_extractor()
        ex.set_num_threads(4)
        
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
        detections = postprocess_yolo11(
            frame, 
            outputs, 
            CONFIG["conf_threshold"], 
            CONFIG["nms_threshold"],
            num_classes
        )
        
        # Vẽ detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            class_id = det["class_id"]
            
            # Đảm bảo class_id hợp lệ
            if 0 <= class_id < len(CONFIG['class_names']):
                label = f"{CONFIG['class_names'][class_id]}: {score:.2f}"
            else:
                label = f"class_{class_id}: {score:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate and draw FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        try:
            result_queue.put(frame, timeout=0.1)
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

    # 5. Giữ main thread sống
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        picam2.stop()
        print("[INFO] Done.")