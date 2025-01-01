from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *
import threading
import queue
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low

import time

app = Flask(__name__)

# Create separate queues for each video feed
frame_queues = [queue.Queue(maxsize=10) for _ in range(4)]
data_queues = [queue.Queue(maxsize=1) for _ in range(4)]

class TrafficData(Model):
    feed_id: int
    vehicle_count: int
    green_time: int
    signal_state: str

class VehicleDetector:
    def __init__(self):
        self.model = YOLO("yolov8l.pt")
        self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.1)
        self.total_count = []

        # Constants
        self.MAX_SIGNAL_TIME = 120
        self.MIN_SIGNAL_TIME = 30
        self.MAX_TRAFFIC = 50
        self.YELLOW_TIME = 5

        # Class names for detection
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
        self.target_classes = ["car", "truck", "bus", "motorbike"]

        # Detection lines (will be initialized with frame dimensions)
        self.limit_lines = None
        self.vehicle_counts_history = []

        # Traffic signal state
        self.signal_state = "GREEN"
        self.signal_start_time = time.time()
        self.current_green_time = 30
        self.vehicles_passed = 0

    def initialize_lines(self, frame):
        height, width = frame.shape[:2]
        y1 = height * 0.6
        y2 = y1 + 20

        self.limit_lines = [
            [int(width * 0.2), int(y1), int(width * 0.8), int(y1)],
            [int(width * 0.2), int(y2), int(width * 0.8), int(y2)]
        ]

    def calculate_green_time(self, vehicle_count):
        density_factor = self.MAX_SIGNAL_TIME / self.MAX_TRAFFIC
        green_time = max(self.MIN_SIGNAL_TIME, min(density_factor * vehicle_count, self.MAX_SIGNAL_TIME))
        return int(green_time)

    def update_signal_state(self):
        current_time = time.time()
        elapsed_time = current_time - self.signal_start_time

        if self.signal_state == "GREEN":
            if self.vehicles_passed >= 2 and elapsed_time >= self.current_green_time:
                self.signal_state = "YELLOW"
                self.signal_start_time = current_time
        elif self.signal_state == "YELLOW":
            if elapsed_time >= self.YELLOW_TIME:
                self.signal_state = "RED"
                self.signal_start_time = current_time
        elif self.signal_state == "RED":
            if elapsed_time >= 5:  # Red light duration
                self.signal_state = "GREEN"
                self.signal_start_time = current_time
                self.vehicles_passed = 0
                self.current_green_time = self.calculate_green_time(len(self.total_count))

    def process_frame(self, frame):
        if self.limit_lines is None:
            self.initialize_lines(frame)

        results = self.model(frame, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if current_class in self.target_classes and conf > 0.3:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        tracked_objects = self.tracker.update(detections)

        for limit in self.limit_lines:
            cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]),
                     (250, 182, 122), 2)

        for result in tracked_objects:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2,
                              colorR=(111, 237, 235))
            cvzone.putTextRect(frame, f'#{int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=1, offset=10,
                               colorR=(56, 245, 213), colorT=(25, 26, 25))

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

            for limit in self.limit_lines:
                if (limit[0] < cx < limit[2] and
                        limit[1] - 15 < cy < limit[1] + 15 and
                        id not in self.total_count):
                    self.total_count.append(id)
                    cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]),
                             (12, 202, 245), 3)
                    if self.signal_state == "GREEN":
                        self.vehicles_passed += 1

        self.update_signal_state()

        # Display signal state and other information
        signal_color = (0, 255, 0) if self.signal_state == "GREEN" else (0, 255, 255) if self.signal_state == "YELLOW" else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (200, 100), signal_color, -1)
        cvzone.putTextRect(frame, f'Signal: {self.signal_state}', (30, 40),
                           scale=1, thickness=2, offset=5,
                           colorR=signal_color, colorT=(0, 0, 0))
        cvzone.putTextRect(frame, f'Count: {len(self.total_count)}', (30, 70),
                           scale=1, thickness=2, offset=5,
                           colorR=signal_color, colorT=(0, 0, 0))
        
        if self.signal_state == "GREEN":
            cvzone.putTextRect(frame, f'Green Time: {self.current_green_time}s', (30, 100),
                               scale=1, thickness=2, offset=5,
                               colorR=signal_color, colorT=(0, 0, 0))
        elif self.signal_state == "YELLOW":
            remaining_yellow = max(0, self.YELLOW_TIME - (time.time() - self.signal_start_time))
            cvzone.putTextRect(frame, f'Yellow Time: {remaining_yellow:.1f}s', (30, 100),
                               scale=1, thickness=2, offset=5,
                               colorR=signal_color, colorT=(0, 0, 0))
        elif self.signal_state == "RED":
            next_green_time = self.calculate_green_time(len(self.total_count))
            cvzone.putTextRect(frame, f'Next Green: {next_green_time}s', (30, 100),
                               scale=1, thickness=2, offset=5,
                               colorR=signal_color, colorT=(0, 0, 0))

        return frame, len(self.total_count), self.current_green_time, self.signal_state

# Create detector instances for each feed
detectors = [VehicleDetector() for _ in range(4)]

# Initialize uAgent
traffic_agent = Agent(name="traffic_agent", seed="traffic_agent_seed")
fund_agent_if_low(traffic_agent.wallet.address())

# Simulated data storage
traffic_data_storage = []

@traffic_agent.on_interval(period=5.0)
async def store_traffic_data(ctx: Context):
    global traffic_data_storage
    for feed_id in range(4):
        try:
            data = data_queues[feed_id].get_nowait()
            traffic_data = TrafficData(feed_id=feed_id, vehicle_count=data['count'], 
                                       green_time=data['green_time'], signal_state=data['signal_state'])
            
            # Store data in simulated storage
            traffic_data_storage.append(traffic_data)
            
            ctx.logger.info(f"Stored traffic data for feed {feed_id}: {traffic_data}")
        except queue.Empty:
            pass

def video_processing_thread(feed_id):
    # Different video sources for each feed
    video_sources = [
        "static/videos/vehical.mp4",
        "static/videos/c.mp4",
        "static/videos/a.mp4",
        "static/videos/b.mp4"
    ]

    cap = cv2.VideoCapture(video_sources[feed_id])

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame, count, green_time, signal_state = detectors[feed_id].process_frame(frame)

        data = {
            "count": count,
            "green_time": green_time,
            "signal_state": signal_state
        }

        try:
            data_queues[feed_id].put(data, block=False)
        except queue.Full:
            try:
                data_queues[feed_id].get_nowait()
                data_queues[feed_id].put(data, block=False)
            except queue.Empty:
                pass

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        try:
            frame_queues[feed_id].put(frame_bytes, block=False)
        except queue.Full:
            try:
                frame_queues[feed_id].get_nowait()
                frame_queues[feed_id].put(frame_bytes, block=False)
            except queue.Empty:
                pass

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames(feed_id):
    while True:
        frame_bytes = frame_queues[feed_id].get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<int:feed_id>')
def video_feed(feed_id):
    if 0 <= feed_id < 4:
        return Response(generate_frames(feed_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Invalid feed ID", 404

@app.route('/get_data/<int:feed_id>')
def get_data(feed_id):
    if 0 <= feed_id < 4:
        try:
            data = data_queues[feed_id].get_nowait()
            return jsonify(data)
        except queue.Empty:
            return jsonify({"count": 0, "green_time": 30, "signal_state": "GREEN"})
    return jsonify({"error": "Invalid feed ID"}), 404

@app.route('/get_stored_data')
def get_stored_data():
    return jsonify([data.dict() for data in traffic_data_storage])

if __name__ == '__main__':
    # Start video processing threads for each feed
    for i in range(4):
        threading.Thread(target=video_processing_thread, args=(i,), daemon=True).start()

    # Start the uAgent in a separate thread
    threading.Thread(target=traffic_agent.run, daemon=True).start()

    app.run(debug=True, threaded=True)