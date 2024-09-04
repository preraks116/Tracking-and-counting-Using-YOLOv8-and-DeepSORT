import ultralytics
import torch

from ultralytics import YOLO

import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
from queue import Queue
from threading import Thread

# CONSTANTS
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
MAX_QUEUE_SIZE = 30
# Class indices for person, car, truck, bus, and motorcycle in COCO dataset
TARGET_CLASSES = [0, 2, 7, 5, 3]

INPUT_VIDEO_PATH = "data/videos/Rec16-1.mp4"
OUTPUT_VIDEO_PATH = "output/Rec16-1-yolo.mp4"

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

def yolo_process_frames(yolo_frame_queue, yolo_result_queue, yolo_model):
    # Class indices for person, car, truck, bus, and motorcycle in COCO dataset
    yolo_target_classes = TARGET_CLASSES
    
    while True:
        yolo_frame = yolo_frame_queue.get()
        if yolo_frame is None:
            break
        
        # Run YOLOv8 inference with specific classes and confidence threshold
        yolo_results = yolo_model(yolo_frame, classes=yolo_target_classes,
                                  conf=YOLO_CONFIDENCE_THRESHOLD)
        
        yolo_result_queue.put((yolo_frame, yolo_results))
    
    yolo_result_queue.put(None)

def yolo_read_frames(yolo_cap, yolo_frame_queue, yolo_max_queue_size):
    while True:
        if yolo_frame_queue.qsize() < yolo_max_queue_size:
            ret, frame = yolo_cap.read()
            if not ret:
                break
            yolo_frame_queue.put(frame)
        else:
            time.sleep(0.1)
    yolo_frame_queue.put(None)

def yolo_write_video(yolo_result_queue, yolo_out):
    while True:
        yolo_item = yolo_result_queue.get()
        if yolo_item is None:
            break
        yolo_frame, yolo_results = yolo_item
        yolo_annotated_frame = yolo_results[0].plot()
        yolo_out.write(yolo_annotated_frame)

def yolo_process_video(yolo_input_path, yolo_output_path, yolo_model_path="yolov8n.pt"):
    yolo_model = YOLO(yolo_model_path)

    yolo_cap = cv2.VideoCapture(yolo_input_path)
    yolo_width = int(yolo_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yolo_height = int(yolo_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    yolo_fps = int(yolo_cap.get(cv2.CAP_PROP_FPS))

    yolo_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    yolo_out = cv2.VideoWriter(yolo_output_path, yolo_fourcc, yolo_fps, (yolo_width, yolo_height))

    yolo_frame_queue = Queue(maxsize=30)
    yolo_result_queue = Queue(maxsize=30)

    yolo_read_thread = Thread(target=yolo_read_frames, args=(yolo_cap, yolo_frame_queue, 30))
    yolo_process_thread = Thread(target=yolo_process_frames, args=(yolo_frame_queue, yolo_result_queue, yolo_model))
    yolo_write_thread = Thread(target=yolo_write_video, args=(yolo_result_queue, yolo_out))

    yolo_read_thread.start()
    yolo_process_thread.start()
    yolo_write_thread.start()

    yolo_read_thread.join()
    yolo_process_thread.join()
    yolo_write_thread.join()

    yolo_cap.release()
    yolo_out.release()
    cv2.destroyAllWindows()

yolo_process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)