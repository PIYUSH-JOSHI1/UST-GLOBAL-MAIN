import cv2
import numpy as np
import customtkinter as ctk
from ultralytics import YOLO
from sort import *
import time
import PIL.Image, PIL.ImageTk
from collections import Counter

class TrafficFlowMonitor:
    def __init__(self):
        self.setup_ui()
        self.setup_video_processing()

    def setup_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("Traffic Flow Monitor")
        self.window.geometry("1200x800")

        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.left_panel = ctk.CTkFrame(self.main_frame, width=300)
        self.left_panel.pack(side="left", fill="y", padx=10, pady=10)

        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.create_controls()
        self.create_lane_counters()
        self.create_video_display()

    def create_controls(self):
        ctk.CTkLabel(self.left_panel, text="Controls", font=("Helvetica", 20, "bold")).pack(pady=10)
        
        self.video_button = ctk.CTkButton(self.left_panel, text="Load Video", command=self.load_video)
        self.video_button.pack(pady=5, padx=10, fill="x")
        
        self.start_button = ctk.CTkButton(self.left_panel, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=5, padx=10, fill="x")
        
        self.stop_button = ctk.CTkButton(self.left_panel, text="Stop Processing", command=self.stop_processing)
        self.stop_button.pack(pady=5, padx=10, fill="x")

        self.fps_label = ctk.CTkLabel(self.left_panel, text="FPS: 0")
        self.fps_label.pack(pady=5, padx=10)

        self.fps_slider = ctk.CTkSlider(self.left_panel, from_=1, to=30, number_of_steps=29, command=self.update_fps)
        self.fps_slider.pack(pady=5, padx=10, fill="x")
        self.fps_slider.set(15)  # Default to 15 FPS

    def create_lane_counters(self):
        ctk.CTkLabel(self.left_panel, text="Lane Counts", font=("Helvetica", 20, "bold")).pack(pady=(20,10))
        
        self.lane_counts = []
        for i in range(4):
            lane_frame = ctk.CTkFrame(self.left_panel)
            lane_frame.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(lane_frame, text=f"Lane {i+1}:").pack(side="left", padx=(0,10))
            count_label = ctk.CTkLabel(lane_frame, text="0")
            count_label.pack(side="right")
            self.lane_counts.append(count_label)

    def create_video_display(self):
        self.video_label = ctk.CTkLabel(self.right_panel, text="")
        self.video_label.pack(fill="both", expand=True)

    def setup_video_processing(self):
        self.model = YOLO("yolov8l.pt")
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]

        self.limits = [
            ([935, 90, 1275, 90], [935, 110, 1275, 110]),
            ([1365, 120, 1365, 360], [1385, 120, 1385, 360]),
            ([600, 70, 600, 170], [620, 70, 620, 170]),
            ([450, 500, 1240, 500], [450, 520, 1240, 520])
        ]

        self.totalCounts = [[] for _ in range(4)]
        self.processing = False
        self.target_fps = 15
        self.last_frame_time = 0

    def load_video(self):
        video_path = ctk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.video_button.configure(text=f"Loaded: {video_path.split('/')[-1]}")

    def start_processing(self):
        if hasattr(self, 'cap'):
            self.processing = True
            self.process_video()
        else:
            ctk.messagebox.showerror("Error", "Please load a video first.")

    def stop_processing(self):
        self.processing = False

    def update_fps(self, value):
        self.target_fps = int(value)
        self.fps_label.configure(text=f"Target FPS: {self.target_fps}")

    def process_video(self):
        if self.processing and hasattr(self, 'cap'):
            current_time = time.time()
            elapsed_time = current_time - self.last_frame_time

            if elapsed_time > 1.0 / self.target_fps:
                success, img = self.cap.read()
                if success:
                    img = self.process_frame(img)
                    self.update_display(img)
                    self.last_frame_time = current_time
                else:
                    self.cap.release()
                    self.processing = False
                    self.video_button.configure(text="Load Video")
                    return

            self.window.after(1, self.process_video)

    def process_frame(self, img):
        imgRegion = cv2.bitwise_and(img, img)
        results = self.model(imgRegion, stream=True)
        detections = np.empty((0, 5))
        detected_vehicles = Counter()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                currentClass = self.classNames[cls]

                if currentClass in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
                    detected_vehicles[currentClass] += 1

        resultsTracker = self.tracker.update(detections)

        for limit_pair, count in zip(self.limits, self.totalCounts):
            cv2.line(img, (limit_pair[0][0], limit_pair[0][1]), (limit_pair[0][2], limit_pair[0][3]), (250, 182, 122), 2)
            cv2.line(img, (limit_pair[1][0], limit_pair[1][1]), (limit_pair[1][2], limit_pair[1][3]), (250, 182, 122), 2)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (111, 237, 235), 2)
            cv2.putText(img, f'{int(id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 245, 213), 2)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

            for i, (limit1, limit2) in enumerate(self.limits):
                if self.check_crossing(cx, cy, limit1, limit2):
                    if id not in self.totalCounts[i]:
                        self.totalCounts[i].append(id)

        for i, count in enumerate(self.totalCounts):
            self.lane_counts[i].configure(text=str(len(count)))

        # Display detected vehicle types at the top of the frame
        vehicle_text = ", ".join([f"{vehicle}: {count}" for vehicle, count in detected_vehicles.items()])
        cv2.putText(img, vehicle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

    def check_crossing(self, cx, cy, limit1, limit2):
        if len(limit1) == 4:  # Horizontal line
            return (limit1[0] < cx < limit1[2] and 
                    limit1[1] - 15 < cy < limit2[1] + 15)
        else:  # Vertical line
            return (limit1[1] < cy < limit1[3] and 
                    limit1[0] - 15 < cx < limit2[0] + 15)

    def update_display(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TrafficFlowMonitor()
    app.run()

