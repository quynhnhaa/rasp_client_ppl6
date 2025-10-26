import time
from multiprocessing import Manager, Lock
from ctypes import c_char_p
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import paho.mqtt.client as mqtt
from sqlalchemy.orm import Session
from app.crud import products as crud_products
from app import models
from app.database import get_db


# Import các modules của ứng dụng
from collections import Counter
import asyncio
import json

# ========== MQTT Setup ==========
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pbl6/products"

mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected successfully to broker.")
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

try:
    mqtt_client.on_connect = on_connect
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"[ERROR] Could not start MQTT client: {e}")
# ==============================


router = APIRouter(prefix="/stream", tags=["Stream"])
def detect_segments(frames_labels, min_detect=60, min_silence=15):
    """
    frames_labels: list of frame label lists, ví dụ [[{label:..}], [], [], ...]
    min_detect: số frame tối thiểu để xem là 1 khoảng detect
    min_silence: số frame tối thiểu để xem là 1 khoảng lặng
    """
    if not frames_labels:
        return []

    segments = []
    current_type = "detect" if len(frames_labels[0]) > 0 else "silence"
    start_idx = 0

    # --- 1️⃣ Tạo danh sách các khoảng thô ---
    for i in range(1, len(frames_labels)):
        this_type = "detect" if len(frames_labels[i]) > 0 else "silence"
        if this_type != current_type:
            segments.append({
                "type": current_type,
                "start": start_idx,
                "end": i - 1,
                "frames": frames_labels[start_idx:i]
            })
            start_idx = i
            current_type = this_type

    # Thêm khoảng cuối cùng
    segments.append({
        "type": current_type,
        "start": start_idx,
        "end": len(frames_labels) - 1,
        "frames": frames_labels[start_idx:]
    })

    # --- 2️⃣ Gộp các khoảng nhỏ hơn ngưỡng ---
    merged = []
    for seg in segments:
        seg_len = seg["end"] - seg["start"] + 1
        too_short = (
            (seg["type"] == "detect" and seg_len < min_detect) or
            (seg["type"] == "silence" and seg_len < min_silence)
        )

        if too_short and merged:
            # Gộp vào khoảng trước
            merged[-1]["end"] = seg["end"]
            merged[-1]["frames"].extend(seg["frames"])
        else:
            merged.append(seg)

    # --- 3️⃣ Gộp các khoảng cùng loại liền nhau ---
    final_segments = []
    for seg in merged:
        if final_segments and seg["type"] == final_segments[-1]["type"]:
            final_segments[-1]["end"] = seg["end"]
            final_segments[-1]["frames"].extend(seg["frames"])
        else:
            final_segments.append(seg)

    return final_segments

from collections import Counter

def choose_representative_frames(frames_labels, segments):
    results = []

    for seg in segments:
        if seg["type"] != "detect":
            continue  # chỉ xử lý khoảng detect

        start, end = seg["start"], seg["end"]
        segment_frames = frames_labels[start:end + 1]

        # Tính độ trùng lặp (độ phổ biến nhãn) cho từng frame
        frame_scores = []
        for frame_idx, frame_labels in enumerate(segment_frames):
            labels = [item["label"] for item in frame_labels]
            # Đếm số lượng nhãn trùng nhau trong toàn đoạn
            total_labels_in_segment = [
                item["label"]
                for frame in segment_frames
                for item in frame
            ]
            overlap_count = sum(lbl in labels for lbl in total_labels_in_segment)
            frame_scores.append((frame_idx, overlap_count))

        # Tìm frame có độ trùng lặp cao nhất
        best_frame_idx, _ = max(frame_scores, key=lambda x: x[1], default=(None, 0))

        if best_frame_idx is not None:
            representative_frame = segment_frames[best_frame_idx]
            results.append({
                "segment": seg,
                "representative_labels": representative_frame
            })

    return results


def generate_mjpeg_stream(shared_frame_buffer, frame_lock):
    mjpeg_header = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    while True:
        with frame_lock:
            jpg_buffer = shared_frame_buffer.value
        if jpg_buffer:
            yield mjpeg_header
            yield jpg_buffer
            yield b'\r\n'
        time.sleep(0.033)  # ~30 FPS

async def event_generator(detected_labels_history, frame_lock_labels):
    last_data = None
    while True:
        await asyncio.sleep(0.04)  # 25 lần/giây ≈ 40ms
        with frame_lock_labels:
            data = list(detected_labels_history)

        if data != last_data or True:  # chỉ gửi khi có thay đổi
            last_data = data
            yield f"data: {json.dumps(data)}\n\n"
from collections import Counter
import json
import time

def process_segments(frames_labels, silence_threshold=20):
    """
    frames_labels: list[list[dict]]
        - mỗi phần tử là danh sách các nhãn detect được của 1 frame
          ví dụ: [[{"label": "Pepsi", "quantity": 1, "time": ...}], [], ...]
    """

    # ============================================================
    # 1️⃣ Xác định các khoảng detect / silence ban đầu
    # ============================================================
    segments = []
    current_type = None
    start = 0

    for i, frame in enumerate(frames_labels):
        frame_type = "silence" if len(frame) == 0 else "detect"

        if current_type is None:
            current_type = frame_type
            start = i
        elif frame_type != current_type:
            segments.append({"type": current_type, "start": start, "end": i - 1})
            current_type = frame_type
            start = i

    if current_type is not None:
        segments.append({"type": current_type, "start": start, "end": len(frames_labels) - 1})

    # ============================================================
    # 2️⃣ Các khoảng lặng nhỏ hơn threshold => chuyển sang detect
    # ============================================================
    for seg in segments:
        if seg["type"] == "silence" and (seg["end"] - seg["start"] + 1) < silence_threshold:
            seg["type"] = "detect"

    # ============================================================
    # 3️⃣ Gộp các khoảng kề nhau có cùng loại
    # ============================================================
    merged_segments = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last = merged_segments[-1]
            if last["type"] == seg["type"]:
                last["end"] = seg["end"]
            else:
                merged_segments.append(seg)

    # ============================================================
    # 4️⃣ Chọn frame trùng lặp nhiều nhất trong từng khoảng detect
    # ============================================================
    representative_frames = []
    for seg in merged_segments:
        if seg["type"] != "detect":
            continue

        start, end = seg["start"], seg["end"]
        segment_frames = frames_labels[start:end + 1]

        # Chuẩn hóa mỗi frame để so sánh (label + quantity)
        frame_serialized = [
            json.dumps(sorted(
                [{"label": item["label"], "quantity": item["quantity"]} 
                 for item in frame],
                key=lambda x: x["label"]
            ))
            for frame in segment_frames
        ]

        frame_counter = Counter(frame_serialized)
        most_common_frames = frame_counter.most_common()

        # Chọn frame phổ biến nhất (bỏ frame rỗng nếu có)
        best_frame = None
        for frame_json, _ in most_common_frames:
            frame_obj = json.loads(frame_json)
            if len(frame_obj) > 0:
                best_frame = frame_obj
                break

        if best_frame is None:
            continue

        representative_frames.append({
            "segment": seg,
            "representative_labels": best_frame
        })

    # ============================================================
    # 5️⃣ Gộp tất cả frame đại diện thành mảng nhãn tổng hợp
    # ============================================================
    total_labels = {}
    for rep in representative_frames:
        for item in rep["representative_labels"]:
            lbl = item["label"]
            qty = item["quantity"]
            total_labels[lbl] = total_labels.get(lbl, 0) + qty

    total_labels_array = [{"label": lbl, "quantity": qty} for lbl, qty in total_labels.items()]

    # ============================================================
    return merged_segments, representative_frames, total_labels_array


@router.get("/video_feed")
async def video_feed_endpoint(request: Request):
    buffer = request.app.state.detected_frame_buffer
    lock = request.app.state.frame_lock_detect
    return StreamingResponse(
        generate_mjpeg_stream(buffer, lock),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
# --- Endpoint stream dữ liệu ---
@router.get("/label_feed")
async def label_feed(request: Request):
    buffer = request.app.state.detected_labels_history
    lock = request.app.state.detected_labels_lock
    return StreamingResponse(event_generator(buffer, lock), media_type="text/event-stream")

@router.get("/product_feed")
async def label_feed(request: Request, db: Session = Depends(get_db)):
    buffer = request.app.state.detected_labels_history
    lock = request.app.state.detected_labels_lock
    async def merge_label_generate(buffer, lock, db_session):
        detected_label = []
        current_time = time.time()
        last_sent_len = -1 # Track the length of last sent frames

        while True:
            await asyncio.sleep(0.04)
            with lock:
                detected_label = list(buffer)
            if time.time() - current_time >= 1:
                current_time = time.time()
                merged_segments, representative_frames, total_labels_array = process_segments(detected_label)
                
                # --- MQTT Publishing Logic ---
                # Only publish if the number of representative frames has changed
                if len(representative_frames) != last_sent_len:
                    print(f"[MQTT] Frame count changed from {last_sent_len} to {len(representative_frames)}. Preparing to publish.")
                    try:
                        if len(representative_frames) >= 2:
                            # Get the second to last representative frame as originally requested
                            target_frame = representative_frames[-2]
                            labels_in_frame = target_frame.get("representative_labels", [])
                            
                            for item in labels_in_frame:
                                product_code = item.get("label")
                                quantity = item.get("quantity")
                                
                                if product_code:
                                    product_db = crud_products.get_product_by_code(db=db_session, code=product_code)
                                    if product_db:
                                        payload = {
                                            "label": product_db.code,
                                            "price": product_db.price,
                                            "quantity": quantity
                                        }
                                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                                        print(f"[MQTT] Published on len change: {payload}")
                                    else:
                                        print(f"[DB] Product with code '{product_code}' not found.")
                    except Exception as e:
                        print(f"[ERROR] Failed during MQTT publish process: {e}")
                    finally:
                        # Update the last sent length after processing
                        last_sent_len = len(representative_frames)

                yield f"data: {json.dumps({'merged_segments': merged_segments, 'representative_frames': representative_frames, 'total_labels_array': total_labels_array})}\n\n"

        
    return StreamingResponse(merge_label_generate(buffer, lock, db), media_type="text/event-stream")

@router.get("/product_feedd")
async def label_feed(request: Request):
    buffer = request.app.state.detected_labels_history
    lock = request.app.state.detected_labels_lock
    async def merge_label_generate(buffer, lock):
        detected_label = []
        while True:
            await asyncio.sleep(1)
            with lock:
                detected_label.append(list(buffer))
            segment_label = detect_segments(detected_label)
            result = choose_representative_frames(frames_labels=detected_label, segments=segment_label)
            yield f"data: {json.dumps(result)}\n\n"

        
        
    return StreamingResponse(merge_label_generate(buffer, lock), media_type="text/event-stream")
