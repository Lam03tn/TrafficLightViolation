import random
import time
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
import torch
from threading import Thread
from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy as np
from collections import deque

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
object_counter = {}
object_counter1 = {}

class TrafficControlApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Control System")
        
        # Initialize variables
        self.detection_active = False
        self.line_start = None
        self.line_end = None
        self.line_scaled = None
        self.video_thread = None
        self.cap = None
        self.running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, padx=5)
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.select_video).grid(row=0, column=2, padx=5)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        style = ttk.Style()
        style.configure("Red.TButton", foreground="red")
        style.configure("Green.TButton", foreground="green")
        
        ttk.Button(control_frame, text="RED", style="Red.TButton",
                  command=lambda: self.toggle_detection(True)).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="GREEN", style="Green.TButton",
                  command=lambda: self.toggle_detection(False)).grid(row=0, column=1, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Status: Waiting for video")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Start button
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=3, column=0, columnspan=2, pady=5)
        
    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set("Status: Video selected")
    
    def toggle_detection(self, state):
        self.detection_active = state
        status = "RED" if state else "GREEN"
        self.status_var.set(f"Traffic signal: {status}")
    
    def select_line(self, frame):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.line_start is None:
                    self.line_start = (x, y)
                elif self.line_end is None:
                    self.line_end = (x, y)
        
        clone = frame.copy()
        cv2.namedWindow("Select Line Points")
        cv2.setMouseCallback("Select Line Points", mouse_callback)
        
        while True:
            display = clone.copy()
            if self.line_start:
                cv2.circle(display, self.line_start, 5, (0, 255, 0), -1)
            if self.line_start and self.line_end:
                cv2.line(display, self.line_start, self.line_end, (0, 255, 0), 2)
                cv2.putText(display, "Press ENTER to continue", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Select Line Points", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 and self.line_start and self.line_end:  # Enter key
                break
            elif key == 27:  # Escape key
                self.line_start = None
                self.line_end = None
                break
        
        cv2.destroyWindow("Select Line Points")
        return self.line_start, self.line_end
    
    def init_tracker(self):
        global deepsort
        deepsort = DeepSort(
            max_cosine_distance=0,
            nms_max_overlap=0.5,
            max_iou_distance=0.9,
            max_age=1,
            n_init=3,
            nn_budget=100,
        )

    def compute_color_for_labels(self, label):
        label = int(label)
        if label == 0:  # Car
            color = (85, 45, 255)
        elif label == 1:  # Motro
            color = (222, 82, 175)
        elif label == 2:  # Bus
            color = (0, 204, 255)
        elif label == 3:  # Bicycle
            color = (0, 149, 255)
        else:
            color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Drawing code from original implementation
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
        
        cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
        cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
        cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
        cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
        
        return img

    def UI_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            img = self.draw_border(img, (c1[0], c1[1] - t_size[1] - 3), 
                                 (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], 
                       thickness=tf, lineType=cv2.LINE_AA)

    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def get_direction(self, point1, point2):
        direction_str = ""
        if point1[1] > point2[1]:
            direction_str += "South"
        elif point1[1] < point2[1]:
            direction_str += "North"
            
        if point1[0] > point2[0]:
            direction_str += "East"
        elif point1[0] < point2[0]:
            direction_str += "West"
            
        return direction_str

    def draw_boxes(self, img, bbox, names, object_id, identities=None, offset=(0, 0), mode='off'):
        if mode == 'off':
            cv2.line(img, self.line_scaled[0], self.line_scaled[1], (0, 255, 0), 3)
        else:
            cv2.line(img, self.line_scaled[0], self.line_scaled[1], (0, 0, 255), 3)

        img_copy = img.copy()
        height, width, _ = img.shape

        for key in list(data_deque):
            if key not in identities:
                data_deque.pop(key)

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            center = (int((x2 + x1) / 2), int((y1 + y2) / 2))
            id = int(identities[i]) if identities is not None else 0

            if id not in data_deque:
                data_deque[id] = deque(maxlen=64)

            color = self.compute_color_for_labels(object_id[i])
            obj_name = names[object_id[i]]
            label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

            data_deque[id].appendleft(center)

            if mode == 'on':
                if len(data_deque[id]) >= 2:
                    direction = self.get_direction(data_deque[id][0], data_deque[id][1])
                    if self.intersect(data_deque[id][0], data_deque[id][1], 
                                    self.line_scaled[0], self.line_scaled[1]):
                        if "North" in direction:
                            cv2.line(img, self.line_scaled[0], self.line_scaled[1], 
                                   (255, 255, 255), 3)
                            cv2.line(img, data_deque[id][0], data_deque[id][1], 
                                   (0, 0, 0), 3)
                            if obj_name not in object_counter1:
                                object_counter1[obj_name] = 1
                            else:
                                object_counter1[obj_name] += 1
                            self.UI_box(box, img_copy, label=label, color=color, line_thickness=2)
                            cv2.imshow("Fault vehicle", img_copy)
                            if 0xFF == ord('q'):
                                break
                            # if cv2.waitKey(33) & (cv2.getWindowProperty("Fault vehicle", cv2.WND_PROP_VISIBLE) < 1 or 0xFF == ord('q')) or self.running == False:
                            #     break

            self.UI_box(box, img, label=label, color=color, line_thickness=2)

        return img

    def process_frame(self, frame, model=None, deepsort=None, mode='off'):
        if model is None or deepsort is None:
            return frame

        results = model(frame, verbose=False)
        bbox_xyxy = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()

        indices = np.where(scores > 0.7)[0]
        bbox_xyxy = bbox_xyxy[indices]
        scores = scores[indices]
        labels = labels[indices]

        deepsort_detections = [
            ([box[0], box[1], box[2] - box[0], box[3] - box[1]], score, label)
            for box, score, label in zip(bbox_xyxy, scores, labels)
        ]

        tracks = deepsort.update_tracks(deepsort_detections, frame=frame)
        identities = [int(track.track_id) for track in tracks]
        bbox = [track.to_tlbr() for track in tracks]
        object_classes = [track.det_class for track in tracks]

        frame = self.draw_boxes(frame, bbox, model.names, object_classes, identities, mode=mode)
        return frame

    
    def run_video(self, weights="models/yolov5.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(weights).to(device)
        self.init_tracker()
        
        self.cap = cv2.VideoCapture(self.file_path_var.get())
        ret, first_frame = self.cap.read()
        if ret:
            # Get line coordinates
            self.line_start, self.line_end = self.select_line(first_frame)
            self.line_scaled = [self.line_start, self.line_end]
            
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.detection_active:
                    frame = self.process_frame(frame, model,deepsort, mode='on')
                else:
                    frame = self.process_frame(frame, model,deepsort, mode='off')
                
                # Show status
                status = "RED" if self.detection_active else "GREEN"
                color = (0, 0, 255) if self.detection_active else (0, 255, 0)
                cv2.putText(frame, f"Traffic signal: {status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.imshow("Traffic Control", frame)
                while True:
                    key = cv2.waitKey(1) & 0xFF  # Wait for 1ms and check for key press

                    # If 'Esc' is pressed, close the window
                    if key == 27:  # 27 is the ord value for the Escape key
                        break

                    # If the window is closed, exit the loop
                    if cv2.getWindowProperty("Fault vehicle", cv2.WND_PROP_VISIBLE) < 1:
                        break

                    if self.running == False:
                        break

        
        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
    
    def start_processing(self):
        if not self.file_path_var.get():
            self.status_var.set("Status: Please select a video file first")
            return
        
        if not self.running:
            self.running = True
            self.video_thread = Thread(target=self.run_video, daemon=True)
            self.video_thread.start()
            self.start_button.configure(text="Stop Processing", command=self.stop_processing)
        
    def stop_processing(self):
        self.running = False
        time.sleep(7)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.start_button.configure(text="Start Processing", command=self.start_processing)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficControlApp()
    app.run() 