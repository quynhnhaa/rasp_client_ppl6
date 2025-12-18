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

# Cần thêm thư viện numpy nếu chưa có
# import numpy as np 

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
    "camera_resolution": (320, 320), # Độ phân giải camera
    "queue_size": 1,
    "input_size": (320, 320),  # Phải khớp với imgsz khi export sang NCNN
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
    "input_layer": "in0", # Tên lớp input của model NCNN
    "output_layer": "out0", # Tên lớp output của model NCNN
    "num_threads": 4,  # Số threads cho inference
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Hàm load NCNN model
# ---------------------
def load_ncnn_model(model_name):
    # Lấy tên model không có đuôi .pt, ví dụ: "no_mosaic_sgd_ms_07"
    base_model_name = os.path.splitext(model_name)[0]
    export_dir = os.path.join(os.path.dirname(__file__), base_model_name + "_ncnn_model") # Giả định model export nằm cùng cấp
    
    param_path = os.path.join(export_dir, "model.ncnn.param")
    bin_path = os.path.join(export_dir, "model.ncnn.bin")
    metadata_path = os.path.join(export_dir, "metadata.yaml")
    
    CONFIG["class_names"] = load_class_names_from_yaml(metadata_path)

    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(f"Khong tim thay file .param hoac .bin trong: {export_dir}")
    
    net = ncnn.Net()
    
    # Cấu hình NCNN Net
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
    # Thời gian chờ nếu queue đầy
    wait_time_if_full = 0.001 
    
    while True:
        # picamera2.capture_array() trả về ảnh RGB888 (Height, Width, Channel), uint8
        frame = picam2.capture_array() 
        try:
            # Thêm frame vào queue. Nếu queue đầy, bỏ qua frame hiện tại để không blocking
            frame_queue.put(frame, timeout=wait_time_if_full) 
        except queue.Full:
            # print("Frame queue is full, dropping frame.") # Có thể in thông báo debug
            pass

# ---------------------
# Postprocess cho NCNN YOLO
# ---------------------
def postprocess_ncnn(frame, ncnn_detections, conf_threshold, nms_threshold, class_names):
    """
    Hậu xử lý đầu ra của mô hình NCNN YOLO để lấy bounding box và vẽ lên frame.
    
    Args:
        frame (np.array): Frame gốc từ camera.
        ncnn_detections (np.array): Đầu ra của mô hình NCNN sau khi chuyển thành numpy array.
                                 Dự kiến có dạng (num_boxes, 4 + num_classes)
                                 trong đó 4 là (cx, cy, w, h) và num_classes là scores cho từng class.
        conf_threshold (float): Ngưỡng tin cậy.
        nms_threshold (float): Ngưỡng IoU cho Non-Maximum Suppression.
        class_names (list): Danh sách tên các class.
        
    Returns:
        np.array: Frame đã được vẽ các bounding box.
    """
    frame_height, frame_width = frame.shape[:2]
    input_height, input_width = CONFIG["input_size"]

    # Tỷ lệ scale giữa ảnh gốc và ảnh input của model
    x_factor = frame_width / input_width
    y_factor = frame_height / input_height

    boxes = []
    scores = []
    class_ids = []

    # `ncnn_detections` đã có dạng (num_boxes, 4 + num_classes)
    # Duyệt qua từng phát hiện
    for row in ncnn_detections:
        # Lấy điểm tin cậy của các class (từ cột thứ 4 trở đi)
        # Các scores này thường đã được nhân với objectness score
        class_scores = row[4:] 
        class_id = np.argmax(class_scores).item() # Sử dụng np.argmax thay cho torch.argmax
        max_score = class_scores[class_id].item()

        if max_score > conf_threshold:
            # Lấy tọa độ bounding box (cx, cy, w, h)
            cx, cy, w, h = row[:4]

            # Chuyển đổi tọa độ về kích thước ảnh gốc
            # YOLO output (cx, cy, w, h) -> convert to (x, y, w, h) for OpenCV
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(int(class_id))

    # Áp dụng Non-Maximum Suppression để loại bỏ các box trùng lặp
    # cv2.dnn.NMSBoxes mong đợi scores là numpy array
    indices = cv2.dnn.NMSBoxes(boxes, np.array(scores), conf_threshold, nms_threshold)
    
    if len(indices) == 0:
        return frame # Không có box nào sau NMS

    # Vẽ các bounding box đã được lọc
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        
        # Đảm bảo class_id hợp lệ
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
def inference_worker(net, frame_queue: Queue, result_queue: Queue):
    # Thời gian chờ nếu queue đầy
    wait_time_if_full = 0.001 

    while True:
        try:
            frame = frame_queue.get(timeout=1.0) # Đợi frame từ camera
        except queue.Empty:
            # print("Frame queue is empty, waiting...") # Có thể in thông báo debug
            time.sleep(0.01)
            continue
        
        original_frame = frame.copy() # Giữ bản sao của frame gốc để vẽ bounding box lên

        # --- TIỀN XỬ LÝ (Preprocessing) ---
        # 1. Resize ảnh về kích thước input của model
        # `picamera2.capture_array()` trả về ảnh RGB888, nên không cần cvtColor sang RGB nếu input model là RGB
        resized_frame = cv2.resize(frame, CONFIG["input_size"])
        
        # 2. Tạo ncnn.Mat từ numpy array
        # `from_pixels` tạo ncnn.Mat từ dữ liệu pixel, nó mong đợi dữ liệu uint8
        # và sẽ tự động chuyển đổi định dạng HWC sang CHW hoặc các định dạng khác theo config của model NCNN
        # PIXEL_RGB_A (RGBA) | PIXEL_RGBA (RGBA) | PIXEL_BGR (BGR) | PIXEL_BGRA (BGRA) | PIXEL_GRAY (GRAY) | PIXEL_RGB (RGB)
        in_mat = ncnn.Mat.from_pixels(
            resized_frame.data, 
            ncnn.Mat.PixelType.PIXEL_RGB, # Camera của Raspberry Pi trả về RGB888
            CONFIG["input_size"][0], 
            CONFIG["input_size"][1]
        )
        
        # Có thể thêm chuẩn hóa mean/norm nếu model NCNN yêu cầu.
        # Thông thường, các model YOLO export sang NCNN đã có mean=0, norm=1/255.0 trong file .param
        # nên việc này sẽ được xử lý tự động bởi NCNN.
        # Nếu không, bạn sẽ cần làm thủ công:
        # mean_vals = [0., 0., 0.] # Hoặc giá trị mean thực tế
        # norm_vals = [1/255., 1/255., 1/255.] # Hoặc giá trị norm thực tế
        # in_mat.substract_mean_normalize(mean_vals, norm_vals)
        

        # --- SUY LUẬN (Inference) ---
        ex = net.create_extractor()
        ex.input(CONFIG["input_layer"], in_mat) # Đặt input
        
        out_mat = ncnn.Mat()
        ex.extract(CONFIG["output_layer"], out_mat) # Lấy output

        # --- HẬU XỬ LÝ (Postprocessing) ---
        # 1. Chuyển đổi ncnn.Mat output sang numpy array
        # Output của NCNN sẽ là một tensor phẳng. Cần reshape lại cho đúng cấu trúc
        # Dự kiến output là (num_candidate_boxes * (4 + num_classes))
        ncnn_outputs = np.array(out_mat)
        
        num_classes = len(CONFIG["class_names"])
        # Reshape output thành (num_boxes, 4 + num_classes)
        # -1 nghĩa là numpy sẽ tự động tính số hàng dựa trên số cột còn lại
        detections = ncnn_outputs.reshape((-1, 4 + num_classes))

        # 2. Gọi hàm postprocess đã định nghĩa
        annotated_frame = postprocess_ncnn(
            original_frame, 
            detections, 
            CONFIG["conf_threshold"], 
            CONFIG["nms_threshold"], 
            CONFIG["class_names"]
        )

        try:
            result_queue.put(annotated_frame, timeout=wait_time_if_full)
        except queue.Full:
            # print("Result queue is full, dropping result.")
            pass


# ---------------------
# Thread 3: Gửi qua imagezmq
# ---------------------
def sender_worker(result_queue: Queue, server_address: str, camera_name: str = "raspi_cam"):
    print(f"[INFO] Dang ket noi den server tai {server_address}...")
    # Thử kết nối nhiều lần
    sender = None
    max_retries = 5
    for i in range(max_retries):
        try:
            sender = imagezmq.ImageSender(connect_to=server_address)
            print(f"[INFO] Da ket noi den server tai {server_address}.")
            break
        except Exception as e:
            print(f"[WARNING] Khong the ket noi den server (lan {i+1}/{max_retries}): {e}")
            time.sleep(2) # Chờ trước khi thử lại
    
    if sender is None:
        print("[ERROR] Khong the ket noi den server sau nhieu lan thu. Thread sender se dung.")
        return

    first_frame_sent = False
    while True:
        try:
            frame = result_queue.get(timeout=1.0) # Đợi frame đã xử lý
        except queue.Empty:
            # print("Result queue is empty, waiting...") # Có thể in thông báo debug
            time.sleep(0.01)
            continue

        try:
            sender.send_image(camera_name, frame)
        except Exception as e:
            print(f"[ERROR] Loi khi gui frame qua imagezmq: {e}. Dang thu ket noi lai...")
            # Thử tạo lại sender nếu mất kết nối
            sender = imagezmq.ImageSender(connect_to=server_address)
            first_frame_sent = False # Reset cờ để in thông báo kết nối lại
            continue

        if not first_frame_sent:
            print("[INFO] Client da gui frame dau tien toi server thanh cong!")
            first_frame_sent = True


# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    print("="*50)
    print("YOLO NCNN Inference on Raspberry Pi")
    print("="*50)
    
    # 1. Khởi tạo camera
    picam2 = Picamera2()
    # Cấu hình camera để trả về ảnh RGB888
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
    frame_queue = Queue(maxsize=CONFIG["queue_size"]) # Queue cho frame từ camera
    result_queue = Queue(maxsize=CONFIG["queue_size"]) # Queue cho frame đã xử lý

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
            time.sleep(1) # Giữ main thread chạy
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        picam2.stop()
        print("[INFO] Done.")