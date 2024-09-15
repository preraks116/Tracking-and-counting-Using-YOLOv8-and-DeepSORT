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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

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

INPUT_VIDEO_PATH = "data/videos/Rec16-1_trimmed.mp4"
INPUT_TIMESTAMP_PATH = "output/timestamps/Rec16-1_trimmed.txt"

OUTPUT_VIDEO_PATH = "output/videos/Rec16-1-yolo_trimmed2.mp4"
OUTPUT_JSON_PATH = "output/json/Rec16-1_trimmed_yolo2.json"

def get_circle_BB(whole_frame):
    x_circle = 0.0
    y_circle = 0.0
    r = 0.0

    # Convert to grayscale.
    gray = cv2.cvtColor(whole_frame, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 18, maxRadius = 19)

    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        first_circle = detected_circles[0, :][0]
        x_circle, y_circle, r = map(float, first_circle)  # Convert to float

    return x_circle, y_circle, r

def process_frame(frame, frame_number, consistent_radius):
    x, y, radius = get_circle_BB(frame)
    return frame_number, x, y, radius

def detect_gaze_parallel(video_path, num_workers=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate consistent radius from first 10 frames
    radii = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        _, _, radius = get_circle_BB(frame)
        radii.append(radius)

    # Use the most common radius as the consistent radius
    consistent_radius = Counter(radii).most_common(1)[0][0]
    print(f"Consistent gaze circle radius: {consistent_radius}")

    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for frame_number in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            future = executor.submit(process_frame, frame, frame_number, consistent_radius)
            futures.append(future)

        for future in as_completed(futures):
            frame_number, x, y, _ = future.result()
            results[frame_number] = (x, y)

    # Print results in order
    for frame_number in sorted(results.keys()):
        x, y = results[frame_number]
        timestamp = frame_number / fps
        print(f"Frame {frame_number} (Time: {timestamp:.2f}s) - Gaze coordinates: ({x}, {y})")

    cap.release()
    
detect_gaze_parallel(INPUT_VIDEO_PATH)