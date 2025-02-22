import cv2
import numpy as np
import time
import os
import urllib.request
import ssl
from PIL import Image
import customtkinter as ctk

class ObjectDetector:
    def __init__(self):
        """Initialize the ObjectDetector"""
        self.model = None
        # COCO dataset class names
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                       "train", "truck", "boat", "traffic light", "fire hydrant",
                       "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                       "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                       "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                       "skis", "snowboard", "sports ball", "kite", "baseball bat",
                       "baseball glove", "skateboard", "surfboard", "tennis racket",
                       "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                       "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                       "hot dog", "pizza", "donut", "cake", "chair", "couch",
                       "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                       "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                       "toaster", "sink", "refrigerator", "book", "clock", "vase",
                       "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        self.cap = None
        self.conf_threshold = 0.5
        
        # Download and load the model
        self.download_models()
        self.load_model()
        
    def download_models(self):
        """Download the required model files"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # URLs for YOLOv3 model files
        model_files = {
            'models/yolov3.cfg': 
                'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'models/yolov3.weights':
                'https://pjreddie.com/media/files/yolov3.weights',
            'models/coco.names':
                'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }
        
        # Download files if they don't exist
        for file_path, url in model_files.items():
            if not os.path.exists(file_path):
                print(f"Downloading {file_path}...")
                try:
                    with urllib.request.urlopen(url, context=ssl_context) as response:
                        content = response.read()
                        with open(file_path, 'wb') as f:
                            f.write(content)
                    print(f"Downloaded {file_path}")
                except Exception as e:
                    print(f"Error downloading {file_path}: {e}")
                    raise
                    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            print("Loading model files...")
            config_path = 'models/yolov3.cfg'
            weights_path = 'models/yolov3.weights'
            
            if not os.path.exists(config_path):
                raise FileNotFoundError("Config file not found")
            if not os.path.exists(weights_path):
                raise FileNotFoundError("Weights file not found")
                
            # Load YOLO model
            print("Initializing DNN module...")
            self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Set backend and target
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def start_camera(self):
        """Start the webcam capture with error handling"""
        try:
            print("Initializing camera...")
            # Release any existing capture
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            time.sleep(2)  # Allow time for camera to initialize
            
            if not self.cap.isOpened():
                raise ValueError("Could not open webcam")
                
            print("Camera started successfully")
                
        except Exception as e:
            print(f"Camera Error: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise
            
    def stop_camera(self):
        """Release the webcam"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera stopped")
            
    def set_confidence_threshold(self, conf):
        """Set confidence threshold for detection"""
        self.conf_threshold = float(conf)
            
    def get_frame(self):
        """
        Get a frame from webcam and perform detection
        Returns:
            tuple: (processed_frame, detections)
        """
        if self.cap is None or not self.cap.isOpened():
            print("Camera is not initialized")
            return None, []
            
        try:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                return None, []
            
            # Flip frame horizontally for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            (H, W) = frame.shape[:2]
            
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(
                frame,
                1/255.0,
                (416, 416),
                swapRB=True,
                crop=False
            )
            
            # Get output layer names
            layer_names = self.model.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
            
            # Pass the blob through the network and obtain detections
            self.model.setInput(blob)
            outputs = self.model.forward(output_layers)
            
            # Initialize lists for detected objects
            boxes = []
            confidences = []
            class_ids = []
            
            # Process detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.conf_threshold:
                        # Scale bounding box coordinates back relative to image size
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        
                        # Get top-left corner coordinates
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        # Add detection to lists
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maxima suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.3)
            
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    # Get box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                    cv2.putText(frame, text, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    results.append({
                        'class': self.classes[class_ids[i]],
                        'confidence': confidences[i],
                        'box': (x, y, x + w, y + h)
                    })
            
            # Convert frame to RGB for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to CTkImage
            pil_image = Image.fromarray(frame_rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, 
                                   dark_image=pil_image,
                                   size=pil_image.size)
            
            return ctk_image, results
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return None, []
