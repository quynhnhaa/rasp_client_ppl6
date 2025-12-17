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


# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_0284.pt"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (640, 640),
    "queue_size": 1, # GIẢM KÍCH THƯỚC QUEUE để tiết kiệm RAM
    # --- Cấu hình cho model NCNN ---
    "input_size": (640, 640), # Kích thước input của model, giữ 640x640 để có độ chính xác cao
    "conf_threshold": 0.25,   # Ngưỡng tin cậy để giữ lại một box
    "nms_threshold": 0.45,    # Ngưỡng IoU cho Non-Maximum Suppression
    "class_names": load_class_names("class_to_id.txt"), # Tự động tải từ file
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Hàm load NCNN model
# ---------------------
def load_ncnn_model(model_name):
    model_name = CONFIG["model_name"]
    
    if not os.path.isfile(model_name):
        print(f"[ERROR] Khong tim thay file mo hinh: {model_name}")
        raise FileNotFoundError(model_name)
        
    export_dir = os.path.splitext(model_name)[0] + "_ncnn_model"
    param_path = os.path.join(export_dir, "model.ncnn.param")
    bin_path = os.path.join(export_dir, "model.ncnn.bin")

    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(f"Khong tim thay file .param hoac .bin trong: {export_dir}")

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    print(f"[INFO] Da load NCNN model tu: {export_dir}")
    return net

# ---------------------
# Thread 1: Camera
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    while True:
        frame = picam2.capture_array()   # frame là numpy array 
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            # Nếu queue đầy thì bỏ frame này (tránh lag do dồn frame cũ)
            pass

# ---------------------
# Thread 2: Inference
# ---------------------
def postprocess(frame, outputs, conf_threshold, nms_threshold):
    h, w, _ = frame.shape
    boxes, scores, class_ids = [], [], []

    # outputs shape: (num_features, num_boxes)
    for detection in outputs.T:
        print(detection[:10])
        cx, cy, bw, bh = detection[:4]

        obj_conf = detection[4]
        if obj_conf < conf_threshold:
            continue

        class_scores = detection[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]

        score = obj_conf * class_conf
        if score < conf_threshold:
            continue

        # YOLOv11 NCNN output thường là tọa độ theo input size (pixel)
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # Clamp để tránh out-of-bound
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            results.append({
                "box": (x, y, x + w_box, y + h_box),
                "score": scores[i],
                "class_id": class_ids[i]
            })

    return results

def inference_worker(net: ncnn.Net, frame_queue: Queue, result_queue: Queue):
    # FPS calculation variables
    start_time = time.time()
    frame_count = 0
    fps = 0

    input_w, input_h = CONFIG["input_size"]

    while True:
        frame = frame_queue.get()  # block đến khi có frame

        # 1. Pre-processing
        mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_RGB, frame.shape[1], frame.shape[0], input_w, input_h)
        # mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

        # 2. Inference
        with net.create_extractor() as ex:
            ex.input("in0", mat_in)      # SỬA LẠI: "images" -> "in0" (hoặc tên input đúng)
            _, out = ex.extract("out0")  # SỬA LẠI: "output0" -> "out0" (hoặc tên output đúng)
            outputs = np.array(out)

        # 3. Post-processing
        detections = postprocess(frame, outputs, CONFIG["conf_threshold"], CONFIG["nms_threshold"])
        # Tối ưu hóa: Vẽ trực tiếp lên frame gốc để tiết kiệm RAM, không cần copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            class_id = det["class_id"]
            label = f"{CONFIG['class_names'][class_id]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and draw FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        frame = result_queue.get()  # block đến khi có dữ liệu
        # frame ở đây là numpy array (BGR), imagezmq hỗ trợ trực tiếp
        sender.send_image(camera_name, frame)

        if not first_frame_sent:
            print("[INFO] Client da ket noi va gui frame dau tien toi server thanh cong!")
            first_frame_sent = True

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    # 1. Khởi tạo camera
    picam2 = Picamera2()
    # Yêu cầu camera xuất ra định dạng RGB 3 kênh để tránh chuyển đổi sau này
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # 2. Load YOLO NCNN model
    model = load_ncnn_model(CONFIG["model_name"])

    # 3. Tạo queue cho pipeline
    frame_queue = Queue(maxsize=CONFIG["queue_size"])   # giới hạn để tránh tràn RAM
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

    # 5. Giữ main thread sống, có thể thêm xử lý signal/thoát
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        # Có thể thêm code dừng camera, đóng socket, etc.
        picam2.stop()