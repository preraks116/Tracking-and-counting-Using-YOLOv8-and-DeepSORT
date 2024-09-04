import ultralytics
import torch

import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
from queue import Queue
from threading import Thread

class_names = ['person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet',
         'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush']

MAX_QUEUE_SIZE = 30

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [0, 2, 7, 5, 3]  # person, car, truck, bus, motorcycle

DEEP_SORT_MODEL_PATH = "deep_sort/deep/checkpoint/ckpt.t7"
DEEP_SORT_MAX_AGE = 20

VIDEO_PATH = "data/videos/Rec16-1.mp4"
OUTPUT_PATH = "output/Rec16-1_deepsort3.mp4"

def read_frames(cap, frame_queue, max_queue_size):
    while True:
        if frame_queue.qsize() < max_queue_size:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        else:
            time.sleep(0.1)  # Sleep briefly to prevent busy-waiting
    frame_queue.put(None)  # Signal end of video
    
def process_frames(frame_queue, result_queue, model):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(og_frame, device=0, classes=TARGET_CLASSES, conf=YOLO_CONFIDENCE_THRESHOLD)
        
        result_queue.put((og_frame, results))
    result_queue.put(None)  # Signal end of processing

def track_and_visualize(result_queue, output_queue, tracker, class_names):
    unique_track_ids = set()
    while True:
        item = result_queue.get()
        if item is None:
            break
        
        og_frame, results = item
        
        if len(results) == 0:
            # No detections in this frame
            output_queue.put(og_frame)
            continue
        
        result = results[0]  # Assuming single image input
        boxes = result.boxes
        cls = boxes.cls.tolist()
        xyxy = boxes.xyxy
        conf = boxes.conf
        xywh = boxes.xywh
        
        if len(cls) == 0:
            # No classes detected in this frame
            output_queue.put(og_frame)
            continue
        
        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh.cpu().numpy()
        
        tracks = tracker.update(bboxes_xywh, conf, og_frame)
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1
            
            color_id = track_id % 3
            color = [(0, 0, 255), (255, 0, 0), (0, 255, 0)][color_id]
            
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            
            # Safely get class name
            class_index = int(cls[track_id % len(cls)]) if cls else 0
            class_name = class_names[class_index] if class_index < len(class_names) else "Unknown"
            
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            unique_track_ids.add(track_id)
        
        output_queue.put(og_frame)
    output_queue.put(None)  # Signal end of tracking

def write_video(output_queue, out):
    while True:
        frame = output_queue.get()
        if frame is None:
            break
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def process_video(input_path, output_path, yolo_model_path=YOLO_MODEL_PATH):
    yolo_model = YOLO(yolo_model_path)
    tracker = DeepSort(model_path=DEEP_SORT_MODEL_PATH, max_age=DEEP_SORT_MAX_AGE)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    output_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    
    read_thread = Thread(target=read_frames, args=(cap, frame_queue, MAX_QUEUE_SIZE))
    process_thread = Thread(target=process_frames, args=(frame_queue, result_queue, yolo_model))
    track_thread = Thread(target=track_and_visualize, args=(result_queue, output_queue, tracker, class_names))
    write_thread = Thread(target=write_video, args=(output_queue, out))
    
    read_thread.start()
    process_thread.start()
    track_thread.start()
    write_thread.start()

    read_thread.join()
    process_thread.join()
    track_thread.join()
    write_thread.join()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

process_video(VIDEO_PATH, OUTPUT_PATH)