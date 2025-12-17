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
# Hàm helper
# ---------------------
def load_class_names(filename="class_to_id.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(class_names)} classes from '{filename}'.")
        return class_names
    except FileNotFoundError:
        print(f"[ERROR] Class file '{filename}' not found.")
        return ["product"]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize và pad ảnh giống như YOLO training.
    """
    shape = img.shape[:2]  # [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    # Divide padding into 2 sides
    dw /= 2
    dh /= 2
    
    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


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
    "input_size": (640, 640),
    "conf_threshold": 0.25,
    "nms_threshold": 0.45,
    "class_names": load_class_names("class_to_id.txt"),
    "num_threads": 4,
    # Tên layer đúng từ file .param
    "input_layer": "in0",
    "output_layer": "out0",  # ĐÂY LÀ TÊN OUTPUT BLOB, KHÔNG PHẢI cat_22
}
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"


# ---------------------
# Load NCNN model
# ---------------------
def load_ncnn_model(model_name):
    if not os.path.isfile(model_name):
        raise FileNotFoundError(model_name)
        
    export_dir = os.path.splitext(model_name)[0] + "_ncnn_model"
    param_path = os.path.join(export_dir, "model.ncnn.param")
    bin_path = os.path.join(export_dir, "model.ncnn.bin")

    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(f"Missing .param or .bin in: {export_dir}")

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = CONFIG["num_threads"]
    
    net.load_param(param_path)
    net.load_model(bin_path)
    
    print(f"[INFO] Loaded NCNN model from: {export_dir}")
    print(f"[INFO] Input layer: {CONFIG['input_layer']}")
    print(f"[INFO] Output layer: {CONFIG['output_layer']}")
    return net


# ---------------------
# Camera worker
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    while True:
        frame = picam2.capture_array()
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass


# ---------------------
# Postprocess YOLO11 NCNN - ĐÃ DECODE
# ---------------------
def postprocess_yolo11_decoded(frame, outputs, conf_threshold, nms_threshold, 
                                num_classes, ratio, pad):
    """
    Post-process YOLO11 NCNN output.
    
    QUAN TRỌNG: Output từ NCNN đã được decode hoàn toàn:
    - Boxes đã được scale về pixel coordinates (trên ảnh 640x640)
    - Class probabilities đã qua sigmoid
    
    Output shape: [8400, 4 + num_classes] hoặc [4 + num_classes, 8400]
    Format: [cx, cy, w, h, class_0, class_1, ..., class_n] (đã scale)
    """
    h, w = frame.shape[:2]
    dw, dh = pad

    boxes = []
    scores = []
    class_ids = []

    # Debug shape
    print(f"[DEBUG] Raw output shape: {outputs.shape}") if len(boxes) == 0 else None

    # Handle output shape
    if len(outputs.shape) == 3:
        outputs = outputs.squeeze(0)
    
    # Transpose nếu cần: [4+nc, N] -> [N, 4+nc]
    # YOLO11 NCNN output thường là [8400, 153] hoặc [153, 8400]
    if outputs.shape[0] == (4 + num_classes):
        outputs = outputs.T
    elif outputs.shape[1] == (4 + num_classes):
        pass  # Đã đúng format
    elif outputs.shape[0] < outputs.shape[1]:
        outputs = outputs.T

    for detection in outputs:
        # Class scores đã qua sigmoid, range [0, 1]
        class_scores = detection[4:4 + num_classes]
        class_id = int(np.argmax(class_scores))
        max_score = float(class_scores[class_id])

        if max_score > conf_threshold:
            # Boxes đã được decode về pixel coordinates trên ảnh 640x640
            # Format: [cx, cy, w, h] hoặc [x1, y1, x2, y2]
            cx, cy, bw, bh = detection[:4]
            
            # Chuyển từ center format sang corner format
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            
            # Scale ngược về tọa độ ảnh gốc
            # 1. Trừ đi padding (vì letterbox thêm padding)
            # 2. Chia cho ratio (vì letterbox scale ảnh)
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio
            
            # Clamp to image bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width > 0 and box_height > 0:
                boxes.append([int(x1), int(y1), int(box_width), int(box_height)])
                scores.append(max_score)
                class_ids.append(class_id)

    # NMS
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
# Inference worker
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
        original_frame = frame.copy()

        # 1. PREPROCESSING VỚI LETTERBOX
        img_letterbox, ratio, pad = letterbox(frame, (input_h, input_w))
        
        # Convert to ncnn Mat (camera đã output RGB888)
        mat_in = ncnn.Mat.from_pixels(
            img_letterbox, 
            ncnn.Mat.PixelType.PIXEL_RGB,
            img_letterbox.shape[1], 
            img_letterbox.shape[0]
        )
        
        # Normalize: pixel / 255.0
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

        # 2. Inference
        ex = net.create_extractor()
        
        ret = ex.input(CONFIG["input_layer"], mat_in)
        if ret != 0:
            print(f"[ERROR] Failed to set input: {ret}")
            continue
        
        # SỬ DỤNG out0 - TÊN OUTPUT BLOB ĐÚNG
        ret, out = ex.extract(CONFIG["output_layer"])
        if ret != 0:
            print(f"[ERROR] Failed to extract output: {ret}")
            continue
            
        outputs = np.array(out)

        # Debug first inference
        if first_inference:
            print(f"[DEBUG] ========================================")
            print(f"[DEBUG] Output shape: {outputs.shape}")
            print(f"[DEBUG] Output dtype: {outputs.dtype}")
            print(f"[DEBUG] Output min/max: {outputs.min():.4f}/{outputs.max():.4f}")
            print(f"[DEBUG] Ratio: {ratio}, Pad: {pad}")
            print(f"[DEBUG] Num classes: {num_classes}")
            print(f"[DEBUG] Expected: [8400, {4 + num_classes}] = [8400, {4 + num_classes}]")
            
            # Kiểm tra một vài detection đầu tiên
            if len(outputs.shape) == 2:
                test_out = outputs.T if outputs.shape[0] < outputs.shape[1] else outputs
                print(f"[DEBUG] Sample detection 0: boxes={test_out[0, :4]}, max_class_score={test_out[0, 4:].max():.4f}")
            print(f"[DEBUG] ========================================")
            first_inference = False

        # 3. Post-processing
        detections = postprocess_yolo11_decoded(
            original_frame,
            outputs, 
            CONFIG["conf_threshold"], 
            CONFIG["nms_threshold"],
            num_classes,
            ratio,
            pad
        )
        
        # Vẽ detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            class_id = det["class_id"]
            
            if 0 <= class_id < len(CONFIG['class_names']):
                label = f"{CONFIG['class_names'][class_id]}: {score:.2f}"
            else:
                label = f"class_{class_id}: {score:.2f}"
            
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        cv2.putText(original_frame, f"FPS: {fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original_frame, f"Det: {len(detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        try:
            result_queue.put(original_frame, timeout=0.1)
        except queue.Full:
            pass


# ---------------------
# Sender worker
# ---------------------
def sender_worker(result_queue: Queue, server_address: str, camera_name: str):
    print(f"[INFO] Connecting to server: {server_address}...")
    sender = imagezmq.ImageSender(connect_to=server_address)
    
    first_frame_sent = False
    while True:
        frame = result_queue.get()
        sender.send_image(camera_name, frame)

        if not first_frame_sent:
            print("[INFO] First frame sent successfully!")
            first_frame_sent = True


# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    print("="*50)
    print("YOLO11 NCNN Inference")
    print("="*50)
    
    # 1. Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print(f"[INFO] Camera started: {CONFIG['camera_resolution']}")

    # 2. Load model
    model = load_ncnn_model(CONFIG["model_name"])
    print(f"[INFO] Classes: {len(CONFIG['class_names'])}")

    # 3. Queues
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    result_queue = Queue(maxsize=CONFIG["queue_size"])

    # 4. Threads
    Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True).start()
    Thread(target=inference_worker, args=(model, frame_queue, result_queue), daemon=True).start()
    Thread(target=sender_worker, args=(result_queue, CONFIG["server_address"], CONFIG["camera_name"]), daemon=True).start()
    
    print("[INFO] Running... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        picam2.stop()