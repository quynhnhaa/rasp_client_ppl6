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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Đường dẫn đến thư mục chứa model NCNN.
    # Thư mục này nên chứa các file .param, .bin và metadata.yaml.
    # Ví dụ: 'no_mosaic_sgd_ms_07_ncnn_model'
    "model_dir": os.getenv("MODEL_DIR", "no_mosaic_sgd_ms_07_ncnn_model"),
    # Tên gốc của file model (không có đuôi .param/.bin)
    "model_name": os.getenv("MODEL_NAME", "model.ncnn"),
    "metadata_filename": "metadata.yaml",

    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "camera_resolution": (320, 320),

    "queue_size": 1,

    # Phải khớp với imgsz khi export sang NCNN
    "input_size": (320, 320),  # (width, height)

    "conf_threshold": 0.45,
    "nms_threshold": 0.45,

    "input_layer": "in0",
    "output_layer": "out0",

    # Số threads CPU cho inference
    "num_threads": 4,
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Thread 1: Camera
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    """
    Đọc frame từ Picamera2 và đưa vào queue.
    Picamera2 với format='RGB888' trả về RGB, ta chuyển sang BGR cho giống OpenCV.
    """
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            frame_queue.put(frame_bgr, timeout=0.1)
        except queue.Full:
            # Bỏ frame nếu queue đầy (tránh lag)
            pass


# ---------------------
# Hàm load NCNN model
# ---------------------
def load_ncnn_model(model_dir: str, model_base_name: str, num_threads: int) -> ncnn.Net:
    """
    Load mô hình NCNN từ .param và .bin.
    - model_dir: Thư mục chứa các file model.
    - model_base_name: Tên gốc của model (ví dụ: 'model.ncnn').
    """
    base_name = os.path.splitext(model_base_name)[0]

    param_path = os.path.join(model_dir, base_name + "ncnn.param")
    bin_path = os.path.join(model_dir, base_name + "ncnn.bin")

    if not os.path.isfile(param_path):
        raise FileNotFoundError(f"Không tìm thấy file param: {param_path}")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file bin: {bin_path}")

    net = ncnn.Net()

    opt = ncnn.Option()
    opt.num_threads = num_threads
    # Nếu Pi của bạn có Vulkan và bạn đã build ncnn với Vulkan thì có thể bật:
    # opt.use_vulkan_compute = True
    net.opt = opt

    net.load_param(param_path)
    net.load_model(bin_path)

    print("[INFO] Đã load NCNN model:")
    print(f"       param: {param_path}")
    print(f"       bin  : {bin_path}")
    return net


# ---------------------
# Hậu xử lý (postprocess) giống code YOLO PyTorch
# ---------------------
def postprocess_ncnn(frame, out_mat: ncnn.Mat,
                     conf_threshold: float,
                     nms_threshold: float,
                     class_names,
                     input_size):
    """
    Hậu xử lý output của mô hình NCNN để lấy bounding box.
    Logic bám sát code YOLO (PyTorch) mà bạn đã test.
    """
    frame_height, frame_width = frame.shape[:2]
    input_width, input_height = input_size  # (w, h)

    # Tỷ lệ scale giữa ảnh gốc và ảnh input
    x_factor = frame_width / input_width
    y_factor = frame_height / input_height

    num_classes = len(class_names)
    expected_dims = 4 + num_classes  # 4 tọa độ + num_classes

    # Chuyển ncnn.Mat sang numpy
    out = np.array(out_mat)
    out = np.squeeze(out)  # bỏ các chiều 1 nếu có

    # In shape để debug một lần (nếu cần)
    if not hasattr(postprocess_ncnn, "_shape_printed"):
        print(f"[DEBUG] NCNN raw output shape: {out.shape}")
        postprocess_ncnn._shape_printed = True

    if out.ndim == 1:
        # Không hợp lệ, không làm gì
        return frame

    # Chuẩn hóa về dạng (num_boxes, 4 + num_classes)
    # Code dưới đây cố gắng bắt chước outputs[0].T trong code PyTorch
    if out.shape[0] == expected_dims:
        # (expected_dims, num_boxes) -> transpose
        detections = out.T  # (num_boxes, expected_dims)
    elif out.shape[-1] == expected_dims:
        # (..., expected_dims) -> reshape về (num_boxes, expected_dims)
        detections = out.reshape(-1, expected_dims)
    else:
        # Thử đoán: chiều nào nhỏ hơn thì là số chiều đặc trưng (dims)
        if out.shape[0] < out.shape[1]:
            detections = out.T
        else:
            detections = out

    boxes = []
    scores = []
    class_ids = []

    for row in detections:
        row = np.asarray(row, dtype=np.float32)
        if row.shape[0] < 5:
            continue

        # Lấy điểm tin cậy của các class (từ cột thứ 4 trở đi)
        class_scores = row[4:]
        if class_scores.size == 0:
            continue

        class_id = int(np.argmax(class_scores))
        max_score = float(class_scores[class_id])

        if max_score > conf_threshold:
            # Lấy tọa độ bounding box (cx, cy, w, h)
            cx, cy, w, h = row[:4]

            # Chuyển đổi tọa độ về kích thước ảnh gốc
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            scores.append(max_score)
            class_ids.append(class_id)

    if not boxes:
        return frame

    # Áp dụng Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return frame

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        if 0 <= class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"ID_{class_id}: {score:.2f}"

        # Vẽ bounding box và label lên ảnh (BGR)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return frame


# ---------------------
# Thread 2: Inference
# ---------------------
def inference_worker(model: ncnn.Net,
                     frame_queue: Queue,
                     result_queue: Queue,
                     config: dict):
    """
    Lấy frame từ frame_queue, chạy inference bằng NCNN,
    hậu xử lý (postprocess) giống code YOLO PyTorch, đưa ảnh đã vẽ box vào result_queue.
    """
    input_size = config["input_size"]
    class_names = config["class_names"]
    conf_threshold = config["conf_threshold"]
    nms_threshold = config["nms_threshold"]
    input_layer = config["input_layer"]
    output_layer = config["output_layer"]
    num_threads = config["num_threads"]

    while True:
        frame = frame_queue.get()  # frame BGR
        frame_for_draw = frame.copy()

        h0, w0 = frame.shape[:2]

        # Tiền xử lý:
        # - Resize về input_size
        # - BGR -> RGB (giống code PyTorch: cv2.cvtColor(BGR, COLOR_BGR2RGB))
        # - Scale về [0,1] bằng substract_mean_normalize
        in_mat = ncnn.Mat.from_pixels_resize(
            frame,  # BGR
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            w0,
            h0,
            input_size[0],
            input_size[1],
        )
        in_mat.substract_mean_normalize(
            [0.0, 0.0, 0.0],
            [1 / 255.0, 1 / 255.0, 1 / 255.0],
        )

        # Suy luận với NCNN
        ex = model.create_extractor()
        ex.set_num_threads(num_threads)
        ex.input(input_layer, in_mat)

        out = ncnn.Mat()
        ex.extract(output_layer, out)

        # Hậu xử lý (giống postprocess trong code PyTorch)
        annotated_frame = postprocess_ncnn(
            frame_for_draw,
            out,
            conf_threshold,
            nms_threshold,
            class_names,
            input_size,
        )

        # Đưa frame đã annotate vào result_queue (nếu đầy thì bỏ bớt)
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
    print("=" * 50)
    print("YOLO11 NCNN Inference on Raspberry Pi")
    print("=" * 50)

    # 1. Khởi tạo camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print(f"[INFO] Camera started with resolution {CONFIG['camera_resolution']}")

    # 2. Load class names & YOLO NCNN model
    model_full_dir = os.path.join(SCRIPT_DIR, CONFIG["model_dir"])
    metadata_full_path = os.path.join(model_full_dir, CONFIG["metadata_filename"])
    CONFIG["class_names"] = load_class_names_from_yaml(metadata_full_path)

    model = load_ncnn_model(
        model_full_dir,
        CONFIG["model_name"],
        CONFIG["num_threads"],
    )

    print(f"[INFO] Number of classes: {len(CONFIG['class_names'])}")

    # 3. Tạo queue cho pipeline
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    result_queue = Queue(maxsize=CONFIG["queue_size"])

    # 4. Khởi chạy 3 thread
    cam_thread = Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True)
    inf_thread = Thread(
        target=inference_worker,
        args=(model, frame_queue, result_queue, CONFIG),
        daemon=True,
    )
    send_thread = Thread(
        target=sender_worker,
        args=(result_queue, CONFIG["server_address"], CONFIG["camera_name"]),
        daemon=True,
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