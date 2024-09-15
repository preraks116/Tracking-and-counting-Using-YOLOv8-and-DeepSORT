import ultralytics
import torch

import cv2
import time
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import json
from ultralytics import YOLO
import numpy as np
import os
from tqdm import tqdm
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: int

@dataclass
class ProcessConfig:
    input_path: str
    video_id: str
    timestamps_path: str
    max_queue_size: int
    output_dir: str
    output_video_path: str
    output_json_path: str
    total_frames: int = 0

@dataclass
class YOLOConfig:
    model_path: str
    confidence_threshold: float
    target_classes: List[int]
    model: Any
    timestamps: List[str]

@dataclass
class YOLOProcessConfig:
    # Fields from ProcessConfig
    input_path: str
    video_id: str
    timestamps_path: str
    max_queue_size: int
    output_dir: str
    output_video_path: str
    output_json_path: str
    total_frames: int = 0
    
    # YOLO-specific field
    yolo_config: YOLOConfig = field(default_factory=dict)

class ProgressTracker:
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.lock = threading.Lock()
        self.pbar = None

    def start(self):
        self.pbar = tqdm(total=self.total, desc=self.desc, position=1, leave=False)

    def update(self, n=1):
        with self.lock:
            if self.pbar:
                self.pbar.update(n)

    def close(self):
        if self.pbar:
            self.pbar.close()

class FileIO:
    @staticmethod
    def read_lines(file_path: str) -> List[str]:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]

    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any]):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


class VideoProcessor(ABC):
    def __init__(self, output_dir: str, default_max_queue_size: int = 30, max_workers: int = None):
        self.output_dir = output_dir
        self.default_max_queue_size = default_max_queue_size
        self.max_workers = max_workers
        self.text_reader = FileIO()

    @abstractmethod
    def create_process_config(self, video_config: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def _process_frames(self, frame_queue: Queue, result_queue: Queue, process_config: Any, results_dict: Dict[str, Any], progress_callback: Callable[[int], None]):
        pass

    @abstractmethod
    def _post_process(self, result_queue: Queue, output_queue: Queue, progress_callback: Callable[[int], None]):
        pass

    def process_single_video(self, process_config: Any, pbar: tqdm):
        cap, video_info = self._initialize_video_capture(process_config.input_path)
        process_config.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        queues = self._create_queues(process_config.max_queue_size)
        results_dict: Dict[str, Any] = {}

        pbar.total = process_config.total_frames
        pbar.set_description(f"Processing {process_config.video_id}")
        pbar.reset()

        def progress_callback(n):
            pbar.update(n)

        threads = self._create_and_start_threads(cap, queues, process_config, results_dict, video_info, progress_callback)
        self._join_threads(threads)

        self._save_results(process_config.output_json_path, results_dict)

        cap.release()
        cv2.destroyAllWindows()

    def process_videos(self, video_configs: List[Dict[str, Any]]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            progress_bars = []
            
            for video_config in video_configs:
                config = self.create_process_config(video_config)
                pbar = tqdm(total=0, desc=f"Initializing {config.video_id}", position=len(progress_bars), leave=True)
                progress_bars.append(pbar)
                futures.append(executor.submit(self.process_single_video, config, pbar))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
            
            for pbar in progress_bars:
                pbar.close()

    def _initialize_video_capture(self, input_path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
        cap = cv2.VideoCapture(input_path)
        video_info = VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS))
        )
        return cap, video_info

    def _create_queues(self, max_queue_size: int) -> Dict[str, Queue]:
        return {
            'frame': Queue(maxsize=max_queue_size),
            'result': Queue(maxsize=max_queue_size),
            'output': Queue(maxsize=max_queue_size)
        }

    def _create_and_start_threads(self, cap: cv2.VideoCapture, queues: Dict[str, Queue], 
                                  process_config: Any, results_dict: Dict[str, Any], 
                                  video_info: VideoInfo, progress_callback: Callable[[int], None]) -> List[Thread]:
        threads = [
            Thread(target=self._read_frames, args=(cap, queues['frame'], process_config.max_queue_size)),
            Thread(target=self._process_frames, args=(queues['frame'], queues['result'], process_config, results_dict, progress_callback)),
            Thread(target=self._post_process, args=(queues['result'], queues['output'], progress_callback)),
            Thread(target=self._write_video, args=(process_config.output_video_path, queues['output'], 
                                                   video_info.fps, video_info.width, video_info.height))
        ]
        for thread in threads:
            thread.start()
        return threads


    def _join_threads(self, threads: List[Thread]):
        for thread in threads:
            thread.join()

    def _read_frames(self, cap: cv2.VideoCapture, frame_queue: Queue, max_queue_size: int):
        while True:
            if frame_queue.qsize() < max_queue_size:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame)
            else:
                time.sleep(0.1)
        frame_queue.put(None)

    def _write_video(self, output_path: str, output_queue: Queue, fps: int, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            frame = output_queue.get()
            if frame is None:
                break
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()

    def _save_results(self, output_json_path: str, results_dict: Dict[str, Any]):
        self.text_reader.write_json(output_json_path, results_dict)
    
    def get_unique_output_dir(self, base_path: str) -> str:
        if not os.path.exists(base_path):
            return base_path
        
        counter = 1
        while True:
            new_path = f"{base_path}_copy{counter}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1