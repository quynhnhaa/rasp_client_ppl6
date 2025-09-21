from inference import get_model
import cv2
import os

# Đường dẫn ảnh
image_file = "/Users/quynhnhaa/Documents/Ahn/Study/Year4.1/PBL6/ImageZQ/Vietnamese Productions Classification/test/images/3029c2ea91aa921f78eb94422600891e_jpg.rf.e3a3ea412713fb17fbf26ce44013d2b4.jpg"
image = cv2.imread(image_file)

# Load model từ Roboflow
model = get_model(
    model_id="vietnamese-productions-classification/19",
    api_key="gMIGrMPVZUiwmUML8smO"
)

# Chạy inference
results = model.infer(image)[0]

# Lấy thông tin prediction
print("=" * 30)
for pred in results.predictions:
    print(f"Class: {pred.class_name}")
    print(f"Confidence: {pred.confidence:.2f}")
    print(f"Bounding box: x={pred.x}, y={pred.y}, w={pred.width}, h={pred.height}")
    print(f"Detection ID: {pred.detection_id}")
    print("-" * 30)

# Nếu muốn lưu lại tất cả kết quả thành list dict để tiện xử lý
detections = [
    {
        "class": pred.class_name,
        "confidence": pred.confidence,
        "bbox": [pred.x, pred.y, pred.width, pred.height],
        "id": pred.detection_id
    }
    for pred in results.predictions
]

print("Tổng số dự đoán:", len(detections))
