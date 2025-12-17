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
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_0284.pt"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (320, 320),
    "queue_size": 2, # GIẢM KÍCH THƯỚC QUEUE để tiết kiệm RAM
    # --- Cấu hình cho model NCNN ---
    "input_size": (320, 320), # Kích thước input của model
    "conf_threshold": 0.25,   # Ngưỡng tin cậy để giữ lại một box
    "nms_threshold": 0.45,    # Ngưỡng IoU cho Non-Maximum Suppression
    "class_names": ["product"], # Thay bằng danh sách tên class của bạn
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
        frame = picam2.capture_array()   # frame là numpy array (RGB/BGR tùy config)
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

    # YOLOv8 NCNN output format: [x, y, w, h, class_prob_0, class_prob_1, ...]
    for detection in outputs.T:
        # Lấy class có xác suất cao nhất
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]

        if max_score > conf_threshold:
            # Chuyển đổi tọa độ từ [center_x, center_y, width, height] về [x1, y1, x2, y2]
            cx, cy, width, height = detection[:4]
            x1 = int((cx - width / 2) * w)
            y1 = int((cy - height / 2) * h)
            x2 = int((cx + width / 2) * w)
            y2 = int((cy + height / 2) * h)

            boxes.append([x1, y1, x2 - x1, y2 - y1]) # cv2.dnn.NMSBoxes expects [x, y, w, h]
            scores.append(float(max_score))
            class_ids.append(class_id)

    # Áp dụng Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            final_boxes.append({
                "box": (x, y, x + w, y + h),
                "score": scores[i],
                "class_id": class_ids[i]
            })
    return final_boxes

def inference_worker(net: ncnn.Net, frame_queue: Queue, result_queue: Queue):
    # FPS calculation variables
    start_time = time.time()
    frame_count = 0
    fps = 0

    input_w, input_h = CONFIG["input_size"]

    while True:
        frame = frame_queue.get()  # block đến khi có frame

        # 1. Pre-processing
        # Frame từ Picamera2 đã là RGB, không cần cvtColor
        img_resized = cv2.resize(frame, (input_w, input_h))
        mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_RGB, frame.shape[1], frame.shape[0], input_w, input_h)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

        # 2. Inference
        with net.create_extractor() as ex:
            ex.input("in0", mat_in)      # SỬA LẠI: "images" -> "in0" (hoặc tên input đúng)
            _, out = ex.extract("out0")  # SỬA LẠI: "output0" -> "out0" (hoặc tên output đúng)
            outputs = np.array(out)

        # 3. Post-processing
        detections = postprocess(frame, outputs, CONFIG["conf_threshold"], CONFIG["nms_threshold"])
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            class_id = det["class_id"]
            label = f"{CONFIG['class_names'][class_id]}: {score:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and draw FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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