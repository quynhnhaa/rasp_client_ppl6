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
        return ["product"]  # Trả về giá trị mặc định nếu có lỗi


# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07"),
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
def load_ncnn_model(model_name, num_threads=4):
    """
    Load NCNN model từ file .param và .bin
    
    Args:
        model_name: Tên model (không có extension)
        num_threads: Số threads cho inference
    
    Returns:
        ncnn.Net object
    """
    # Đường dẫn đến file model NCNN
    # Thông thường khi export từ YOLO sẽ có folder: model_name_ncnn_model/
    model_dir = f"{model_name}_ncnn_model"
    param_path = os.path.join(model_dir, "model.ncnn.param")
    bin_path = os.path.join(model_dir, "model.ncnn.bin")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Không tìm thấy file: {param_path}")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file: {bin_path}")
    
    # Tạo và cấu hình NCNN Net
    net = ncnn.Net()
    
    # Cấu hình options
    net.opt.use_vulkan_compute = False  # Tắt Vulkan (không có GPU trên Pi)
    net.opt.num_threads = num_threads
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_packing_layout = True
    
    # Load model
    ret_param = net.load_param(param_path)
    ret_bin = net.load_model(bin_path)
    
    if ret_param != 0 or ret_bin != 0:
        raise RuntimeError(f"Lỗi khi load model NCNN: param={ret_param}, bin={ret_bin}")
    
    print(f"[INFO] Đã load model NCNN từ '{model_dir}'")
    return net


# ---------------------
# Preprocess cho NCNN
# ---------------------
def preprocess(frame, input_width, input_height):
    """
    Tiền xử lý frame ảnh trước khi đưa vào mô hình NCNN.
    
    Args:
        frame: Ảnh BGR từ camera (numpy array HWC)
        input_width: Chiều rộng input của model
        input_height: Chiều cao input của model
    
    Returns:
        ncnn.Mat đã được chuẩn hóa
    """
    # Resize ảnh về kích thước input của model
    img = cv2.resize(frame, (input_width, input_height))

    # Chuyển đổi từ RGB (từ Picamera2) sang BGR vì ncnn.Mat.from_pixels
    # mặc định xử lý BGR tốt hơn khi không chỉ định rõ.
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Tạo ncnn.Mat từ pixels
    # from_pixels nhận: data, pixel_type, width, height
    mat_in = ncnn.Mat.from_pixels( # Mặc định là PIXEL_BGR
        img_bgr,
        ncnn.Mat.PixelType.PIXEL_BGR,
        input_width, 
        input_height
    )
    
    # Chuẩn hóa: (pixel - mean) * norm
    # Với mean=0 và norm=1/255, kết quả là pixel/255 (chuẩn hóa về [0, 1])
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)
    
    return mat_in


# ---------------------
# Postprocess cho YOLO11
# ---------------------
def postprocess(frame, output, conf_threshold, nms_threshold, class_names, input_width, input_height):
    """
    Hậu xử lý output của mô hình để lấy bounding box.
    
    Args:
        frame: Ảnh gốc (BGR, numpy array)
        output: Output từ model NCNN (numpy array)
        conf_threshold: Ngưỡng confidence
        nms_threshold: Ngưỡng NMS
        class_names: Danh sách tên các class
        input_width: Chiều rộng input của model
        input_height: Chiều cao input của model
    
    Returns:
        Frame đã được vẽ bounding box
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Tỷ lệ scale giữa ảnh gốc và ảnh input
    x_factor = frame_width / input_width
    y_factor = frame_height / input_height

    boxes = []
    scores = []
    class_ids = []

    # Output của YOLO11 có shape (num_features, num_boxes)
    # num_features = 4 (cx, cy, w, h) + num_classes
    # Chuyển vị (transpose) để có shape (num_boxes, num_features)
    # Kiểm tra shape và xử lý phù hợp
    if len(output.shape) == 3:
        # Shape: (1, num_features, num_boxes) -> bỏ batch dimension
        output = output[0]
    
    # Transpose để có shape (num_boxes, num_features)
    detections = output.T

    for row in detections:
        # Lấy điểm tin cậy của các class (từ cột thứ 4 trở đi)
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]

        if max_score > conf_threshold:
            # Lấy tọa độ bounding box (cx, cy, w, h)
            cx, cy, w, h = row[:4]

            # Chuyển đổi từ center format sang corner format
            # và scale về kích thước ảnh gốc
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(int(class_id))

    # Áp dụng Non-Maximum Suppression để loại bỏ các box trùng lặp
    if len(boxes) == 0:
        return frame
        
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    
    if len(indices) == 0:
        return frame

    # Vẽ các bounding box lên ảnh
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        
        # Kiểm tra an toàn để tránh lỗi IndexError
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"ID_{class_id}: {score:.2f}"

        # Vẽ bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Vẽ background cho text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, 
            (x, y - text_height - 10), 
            (x + text_width, y), 
            (0, 255, 0), 
            -1
        )
        
        # Vẽ text
        cv2.putText(
            frame, label, (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    return frame


# ---------------------
# Thread 1: Camera
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    """Thread worker để capture frame từ camera."""
    while True:
        frame = picam2.capture_array()
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            # Queue đầy, bỏ qua frame này
            pass


# ---------------------
# Thread 2: Inference
# ---------------------
def inference_worker(net, frame_queue: Queue, result_queue: Queue):
    """
    Thread worker để chạy inference trên các frame.
    
    Args:
        net: NCNN Net object
        frame_queue: Queue chứa frame từ camera
        result_queue: Queue để đưa frame đã xử lý
    """
    input_width, input_height = CONFIG["input_size"]
    conf_threshold = CONFIG["conf_threshold"]
    nms_threshold = CONFIG["nms_threshold"]
    class_names = CONFIG["class_names"]
    input_layer = CONFIG["input_layer"]
    output_layer = CONFIG["output_layer"]
    
    print(f"[INFO] Inference worker started")
    print(f"[INFO] Input size: {input_width}x{input_height}")
    print(f"[INFO] Confidence threshold: {conf_threshold}")
    print(f"[INFO] NMS threshold: {nms_threshold}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Lấy frame từ queue
        frame = frame_queue.get()
        
        
        # 1. Tiền xử lý
        mat_in = preprocess(frame, input_width, input_height)
        
        # 2. Tạo extractor và chạy inference
        ex = net.create_extractor()
        ex.set_light_mode(True)  # Tiết kiệm memory
        ex.input(input_layer, mat_in)
        
        # 3. Lấy output
        ret, mat_out = ex.extract(output_layer)
        
        if ret != 0:
            print(f"[WARNING] Lỗi khi extract output: {ret}")
            continue
        
        # 4. Chuyển đổi output từ ncnn.Mat sang numpy array
        # NCNN Mat có thể có nhiều chiều, cần reshape phù hợp
        output = np.array(mat_out)
        
        # 5. Hậu xử lý - vẽ bounding box
        annotated_frame = postprocess(
            frame.copy(),  # Copy để không ảnh hưởng frame gốc
            output,
            conf_threshold,
            nms_threshold,
            class_names,
            input_width,
            input_height
        )
        
        # Đưa frame đã xử lý vào result queue
        try:
            result_queue.put(annotated_frame, timeout=0.1)
        except queue.Full:
            pass
        
        # Tính FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[INFO] FPS: {fps:.2f}")


# ---------------------
# Thread 3: Gửi qua imagezmq
# ---------------------
def sender_worker(result_queue: Queue, server_address: str, camera_name: str = "raspi_cam"):
    """
    Thread worker để gửi frame qua imagezmq.
    
    Args:
        result_queue: Queue chứa frame đã xử lý
        server_address: Địa chỉ server
        camera_name: Tên camera
    """
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
    print("=" * 50)
    
    # 1. Load class names từ metadata
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(f"{CONFIG["model_name"]}_ncnn_model", "metadata.yaml")
    CONFIG["class_names"] = load_class_names_from_yaml(metadata_path)
    
    # 2. Khởi tạo camera
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    print(f"[INFO] Camera started with resolution {CONFIG['camera_resolution']}")
    
    # Đợi camera ổn định
    time.sleep(2)

    # 3. Load YOLO NCNN model
    model = load_ncnn_model(CONFIG["model_name"], CONFIG["num_threads"])
    print(f"[INFO] Number of classes: {len(CONFIG['class_names'])}")

    # 4. Tạo queue cho pipeline
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    result_queue = Queue(maxsize=CONFIG["queue_size"])

    # 5. Khởi chạy 3 thread
    cam_thread = Thread(
        target=camera_worker, 
        args=(picam2, frame_queue), 
        daemon=True,
        name="CameraThread"
    )
    inf_thread = Thread(
        target=inference_worker, 
        args=(model, frame_queue, result_queue), 
        daemon=True,
        name="InferenceThread"
    )
    send_thread = Thread(
        target=sender_worker,
        args=(result_queue, CONFIG["server_address"], CONFIG["camera_name"]),
        daemon=True,
        name="SenderThread"
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