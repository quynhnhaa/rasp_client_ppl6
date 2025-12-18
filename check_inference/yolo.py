from ultralytics import YOLO
import cv2
import numpy as np
import torch
import yaml
import os

# --- CONFIG ---
MODEL_PATH = "no_mosaic_sgd_ms_07.pt"
VIDEO_PATH = "/Users/quynhnhaa/Documents/Ahn/Study/Year4.1/PBL6/main/rasp_client_ppl6/check_inference/video_20251020_183941.mp4"
METADATA_PATH = "metadata.yaml" # File chứa tên các class

# Tham số của mô hình
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
CONF_THRESHOLD = 0.45  # Ngưỡng tin cậy để lọc bớt các phát hiện
NMS_THRESHOLD = 0.45   # Ngưỡng IoU cho Non-Maximum Suppression

def load_class_names(path):
    """Tải danh sách tên class từ file metadata.yaml."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        # `names` là một dictionary {id: name}, chúng ta cần lấy list các name
        # Sắp xếp theo key (id) để đảm bảo thứ tự đúng
        class_names = [name for _, name in sorted(metadata['names'].items())]
        print(f"[INFO] Đã tải {len(class_names)} class từ '{path}'.")
        return class_names
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file metadata: {e}. Sử dụng class mặc định.")
        return ["product"]

def preprocess(frame, input_width, input_height, device):
    """Tiền xử lý frame ảnh trước khi đưa vào mô hình."""
    # Resize ảnh về kích thước input của model
    img = cv2.resize(frame, (input_width, input_height))
    # Chuyển từ BGR (OpenCV) sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Chuyển layout từ HWC (Height, Width, Channel) sang CHW (Channel, Height, Width)
    img = img.transpose(2, 0, 1)
    # Chuẩn hóa giá trị pixel về [0, 1] và chuyển sang kiểu float32
    img = (img / 255.0).astype(np.float32)
    # Chuyển thành tensor của PyTorch
    tensor = torch.from_numpy(img)
    # Thêm một chiều batch (N) -> NCHW
    tensor = tensor.unsqueeze(0)
    # Chuyển tensor đến device (CPU/GPU)
    return tensor.to(device)

def postprocess(frame, outputs, conf_threshold, nms_threshold, class_names):
    """Hậu xử lý output của mô hình để lấy bounding box."""
    frame_height, frame_width = frame.shape[:2]
    input_height, input_width = INPUT_HEIGHT, INPUT_WIDTH

    # Tỷ lệ scale giữa ảnh gốc và ảnh input
    x_factor = frame_width / input_width
    y_factor = frame_height / input_height

    boxes = []
    scores = []
    class_ids = []

    # Output của YOLOv11n có shape (1, 5, 2100) -> (batch, 4_coords + num_classes, num_boxes)
    # Bỏ chiều batch và chuyển vị (transpose) để có shape (num_boxes, 4_coords + num_classes)
    detections = outputs[0].T

    for row in detections:
        # Lấy điểm tin cậy của các class (từ cột thứ 4 trở đi)
        class_scores = row[4:]
        class_id = torch.argmax(class_scores).item()
        max_score = class_scores[class_id].item()

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
    # Chuyển list sang numpy array để dùng với cv2.dnn.NMSBoxes
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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng device: {device}")

    # 1. Tải model và class names
    model = YOLO(MODEL_PATH).model.to(device).eval() # Lấy model PyTorch gốc, chuyển sang mode eval
    class_names = load_class_names(os.path.join(os.path.dirname(__file__), METADATA_PATH))

    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tắt tính toán gradient để tăng tốc
        with torch.no_grad():
            # 2. Tiền xử lý
            input_tensor = preprocess(frame, INPUT_WIDTH, INPUT_HEIGHT, device)
            # 3. Suy luận
            outputs = model(input_tensor)
            # 4. Hậu xử lý
            # outputs[0] là tensor chứa các bounding box
            annotated_frame = postprocess(frame, outputs[0].cpu(), CONF_THRESHOLD, NMS_THRESHOLD, class_names)

        cv2.imshow("YOLOv11n Manual Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
