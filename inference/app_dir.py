from inference import get_model
import cv2
import os

# Đọc ảnh
input_dir = "images2"
output_dir = "results2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Load model
model = get_model(
    model_id="vietnamese-productions-classification/19",
    api_key="gMIGrMPVZUiwmUML8smO"
)

# Duyet cac file trng thu muc input_dir
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is not None:
            print(f"Processing image: {filename}")
        else:
            print(f"Failed to read image: {filename}")
            continue
    
        # Du doan san pham trong anh
        results = model.infer(image)[0]

        # Vẽ kết quả
        for pred in results.predictions:
            x, y = int(pred.x), int(pred.y)
            w, h = int(pred.width), int(pred.height)
            conf = float(pred.confidence)
            class_name = pred.class_name

            # Tinh toa do cua khung 
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Ve khung chung nha
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Ve nhan + confidence
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, image)
        print(f"Saved result to: {save_path}")

print('Hoan tat')