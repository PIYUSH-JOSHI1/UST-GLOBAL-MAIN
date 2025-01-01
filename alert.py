import cv2
import numpy as np
import customtkinter as ctk
import PIL.Image
import PIL.ImageTk
import os
from datetime import datetime
from ultralytics import YOLO
import time
from sort import *

class AdvancedDetectionSystem:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize main window
        self.window = ctk.CTk()
        self.window.title("Advanced Detection System")
        self.window.geometry("1200x800")

        # Initialize detection models
        self.model = YOLO("yolov8l.pt")
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        # Initialize variables
        self.camera_active = False
        self.current_frame = None
        self.vehicle_detection_active = True
        self.last_time = datetime.now()
        self.frame_count = 0
        self.vehicle_timestamps = {}
        self.alert_threshold = 20  # seconds
        
        # Create UI
        self.create_ui()

    def create_ui(self):
        # Create main container
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Create left panel for controls
        self.left_panel = ctk.CTkFrame(self.main_container, width=250)
        self.left_panel.pack(side="left", fill="y", padx=5, pady=5)

        # Create title in left panel
        ctk.CTkLabel(
            self.left_panel, 
            text="Advanced Detection Controls",
            font=("Helvetica", 18, "bold")
        ).pack(pady=15)

        # Create control buttons
        self.create_control_buttons()

        # Create detection toggles
        self.create_detection_toggles()

        # Create statistics section
        self.create_statistics_section()

        # Create right panel for display
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Create display area
        self.display = ctk.CTkLabel(self.right_panel, text="")
        self.display.pack(padx=10, pady=10, fill="both", expand=True)

        # Create status bar
        self.status_bar = ctk.CTkLabel(
            self.window,
            text="Status: Ready",
            font=("Helvetica", 12)
        )
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)

    def create_control_buttons(self):
        # Button container
        button_frame = ctk.CTkFrame(self.left_panel)
        button_frame.pack(fill="x", padx=10, pady=10)

        # Camera button
        self.camera_button = ctk.CTkButton(
            button_frame,
            text="Start Camera",
            command=self.toggle_camera,
            fg_color="#4CAF50",
            hover_color="#45a049"
        )
        self.camera_button.pack(pady=5, fill="x")

        # Load Image button
        self.load_button = ctk.CTkButton(
            button_frame,
            text="Load Image",
            command=self.load_image,
            fg_color="#2196F3",
            hover_color="#1976D2"
        )
        self.load_button.pack(pady=5, fill="x")

        # Video Upload button
        self.video_button = ctk.CTkButton(
            button_frame,
            text="Upload Video",
            command=self.upload_video,
            fg_color="#9C27B0",
            hover_color="#7B1FA2"
        )
        self.video_button.pack(pady=5, fill="x")

        # Screenshot button
        self.screenshot_button = ctk.CTkButton(
            button_frame,
            text="Save Screenshot",
            command=self.save_screenshot,
            fg_color="#FF9800",
            hover_color="#F57C00"
        )
        self.screenshot_button.pack(pady=5, fill="x")

    def create_detection_toggles(self):
        # Toggle container
        toggle_frame = ctk.CTkFrame(self.left_panel)
        toggle_frame.pack(fill="x", padx=10, pady=10)

        # Vehicle Detection toggle
        ctk.CTkLabel(toggle_frame, text="Vehicle Detection", font=("Helvetica", 14)).pack()
        self.vehicle_toggle = ctk.CTkSwitch(
            toggle_frame,
            text="",
            command=self.toggle_vehicle_detection,
            onvalue=True,
            offvalue=False,
            progress_color="#4CAF50"
        )
        self.vehicle_toggle.pack(pady=5)
        self.vehicle_toggle.select()

    def create_statistics_section(self):
        # Stats container
        stats_frame = ctk.CTkFrame(self.left_panel)
        stats_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            stats_frame,
            text="Detection Statistics",
            font=("Helvetica", 16, "bold")
        ).pack(pady=5)

        # Vehicle count
        self.vehicle_count_label = ctk.CTkLabel(
            stats_frame, 
            text="Vehicles Detected: 0",
            font=("Helvetica", 14)
        )
        self.vehicle_count_label.pack(pady=2)

        # Alert count
        self.alert_count_label = ctk.CTkLabel(
            stats_frame, 
            text="Active Alerts: 0",
            font=("Helvetica", 14)
        )
        self.alert_count_label.pack(pady=2)

        # FPS counter
        self.fps_label = ctk.CTkLabel(
            stats_frame,
            text="FPS: 0",
            font=("Helvetica", 14)
        )
        self.fps_label.pack(pady=2)

    def process_frame(self, frame):
        if not self.vehicle_detection_active:
            return frame

        # Resize frame for processing while maintaining aspect ratio
        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame_resized = cv2.resize(frame, (width, height))

        # Detect vehicles
        results = self.model(frame_resized, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Filter for vehicles (car, truck, bus, motorcycle)
                if cls in [2, 5, 7, 3] and conf > 0.5:
                    # Scale coordinates back to original size
                    x1 = int(x1 * 100 / scale_percent)
                    x2 = int(x2 * 100 / scale_percent)
                    y1 = int(y1 * 100 / scale_percent)
                    y2 = int(y2 * 100 / scale_percent)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Count active alerts
        active_alerts = 0

        # Process tracked objects
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Update timestamp for vehicle
            current_time = time.time()
            if track_id not in self.vehicle_timestamps:
                self.vehicle_timestamps[track_id] = current_time
            
            # Check if vehicle has been stationary
            elapsed_time = current_time - self.vehicle_timestamps[track_id]
            
            if elapsed_time > self.alert_threshold:
                # Draw alert box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                alert_text = f"ALERT! Vehicle {int(track_id)}"
                cv2.putText(frame, alert_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                active_alerts += 1
                
                # Simulate alert to police station
                print(f"Alert sent to police: Stationary vehicle detected at location {x1},{y1}")
            else:
                # Normal tracking box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle {int(track_id)}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update statistics
        self.vehicle_count_label.configure(text=f"Vehicles Detected: {len(tracked_objects)}")
        self.alert_count_label.configure(text=f"Active Alerts: {active_alerts}")
        
        return frame

    def toggle_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Could not open camera")
                return
            self.camera_active = True
            self.camera_button.configure(text="Stop Camera", fg_color="#F44336", hover_color="#D32F2F")
            self.update_camera()
            self.update_status("Camera Active")
        else:
            self.camera_active = False
            self.cap.release()
            self.camera_button.configure(text="Start Camera", fg_color="#4CAF50", hover_color="#45a049")
            self.update_status("Camera Stopped")

    def update_camera(self):
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Convert to PhotoImage
                image = PIL.Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                photo = PIL.ImageTk.PhotoImage(image)
                
                # Update display
                self.display.configure(image=photo)
                self.current_frame = photo
                
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                if hasattr(self, 'last_fps_time'):
                    time_diff = current_time - self.last_fps_time
                    if time_diff >= 1.0:
                        fps = self.frame_count / time_diff
                        self.fps_label.configure(text=f"FPS: {fps:.1f}")
                        self.frame_count = 0
                        self.last_fps_time = current_time
                else:
                    self.last_fps_time = current_time
                
                # Schedule next update (40ms = 25 fps max)
                self.window.after(40, self.update_camera)

    def toggle_vehicle_detection(self):
        self.vehicle_detection_active = self.vehicle_toggle.get()
        status = "enabled" if self.vehicle_detection_active else "disabled"
        self.update_status(f"Vehicle detection {status}")

    def upload_video(self):
        file_path = ctk.filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.show_error("Could not open video file")
            return

        # Reset statistics
        self.vehicle_timestamps.clear()
        self.frame_count = 0
        self.last_fps_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Convert to PhotoImage
            image = PIL.Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            photo = PIL.ImageTk.PhotoImage(image)
            
            # Update display
            self.display.configure(image=photo)
            self.current_frame = photo
            self.window.update()

            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_fps_time
            if time_diff >= 1.0:
                fps = self.frame_count / time_diff
                self.fps_label.configure(text=f"FPS: {fps:.1f}")
                self.frame_count = 0
                self.last_fps_time = current_time

            # Control frame rate
            self.window.after(40)  # 40ms delay for ~25 fps

        cap.release()
        self.update_status("Video processing completed")

    def load_image(self):
        file_path = ctk.filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                self.show_error("Could not load image")
                return
            
            # Process and display image
            processed_image = self.process_frame(image)
            photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            )
            self.display.configure(image=photo)
            self.current_frame = photo
            self.update_status("Image loaded and processed")

    def save_screenshot(self):
        if not hasattr(self, 'current_frame'):
            self.show_error("No image to save")
            return
            
        # Create screenshots directory
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        
        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/detection_{timestamp}.png"
        
        image = PIL.ImageTk.getimage(self.current_frame)
        image.save(filename)
        self.update_status(f"Screenshot saved as {filename}")

    def update_status(self, message):
        self.status_bar.configure(text=f"Status: {message}")

    def show_error(self, message):
        ctk.messagebox.showerror("Error", message)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = AdvancedDetectionSystem()
    app.run()

