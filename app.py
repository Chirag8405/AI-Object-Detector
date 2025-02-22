import time
from detector import ObjectDetector
from ui import ApplicationUI
import tkinter as tk
from tkinter import messagebox

class ObjectDetectionApp:
    def __init__(self):
        """Initialize the application"""
        self.ui = ApplicationUI()
        
        try:
            self.detector = ObjectDetector()
            print("Object detector initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize object detector: {str(e)}")
            self.ui.window.quit()
            return
            
        # Bind the closing event
        self.ui.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind the toggle detection callback
        self.ui.control_button.configure(command=self.toggle_detection)
        
        # Bind confidence threshold update
        self.ui.conf_slider.configure(command=self._update_confidence)
        
    def _update_confidence(self, value):
        """Update detector confidence threshold when slider changes"""
        if hasattr(self, 'detector'):
            self.detector.set_confidence_threshold(float(value))
        
    def toggle_detection(self):
        """Toggle the detection on/off"""
        try:
            if not self.ui.running:  # If currently not running
                # Start detection
                print("Starting camera...")
                self.detector.start_camera()
                self.ui.running = True  # Set running state before updating UI
                self.ui.control_button.configure(
                    text="Stop Detection",
                    fg_color="#c42b1c",
                    hover_color="#a62215"
                )
                self.update_frame()  # Start the update loop
            else:  # If currently running
                # Stop detection
                print("Stopping camera...")
                self.detector.stop_camera()
                self.ui.running = False  # Set running state before updating UI
                self.ui.control_button.configure(
                    text="Start Detection",
                    fg_color=["#3a7ebf", "#1f538d"],
                    hover_color=["#325882", "#14375e"]
                )
                # Clear the video label
                self.ui.video_label.configure(text="Camera Feed Not Started", image="")
                
        except ValueError as e:
            print(f"Camera Error: {e}")
            messagebox.showerror("Camera Error", str(e))
            self.ui.running = False
            self.ui.control_button.configure(
                text="Start Detection",
                fg_color=["#3a7ebf", "#1f538d"],
                hover_color=["#325882", "#14375e"]
            )
        except Exception as e:
            print(f"Unexpected error: {e}")
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")
            self.ui.running = False
            self.ui.control_button.configure(
                text="Start Detection",
                fg_color=["#3a7ebf", "#1f538d"],
                hover_color=["#325882", "#14375e"]
            )
                
    def update_frame(self):
        """Update the frame if detection is running"""
        if self.ui.running:
            try:
                # Get frame and detections
                frame, detections = self.detector.get_frame()
                
                if frame is not None:
                    # Update UI
                    self.ui.update_frame(frame)
                    self.ui.update_detections(detections)
                    
                    # Schedule next update
                    self.ui.window.after(10, self.update_frame)
                else:
                    # Handle camera error
                    print("Failed to get camera frame")
                    self.ui.running = False
                    self.ui.control_button.configure(
                        text="Start Detection",
                        fg_color=["#3a7ebf", "#1f538d"],
                        hover_color=["#325882", "#14375e"]
                    )
                    messagebox.showerror("Error", "Failed to get camera frame. Please check your camera connection.")
            except Exception as e:
                print(f"Error in update_frame: {e}")
                self.ui.running = False
                self.ui.control_button.configure(
                    text="Start Detection",
                    fg_color=["#3a7ebf", "#1f538d"],
                    hover_color=["#325882", "#14375e"]
                )
                messagebox.showerror("Error", f"Error updating frame: {str(e)}")
            
    def on_closing(self):
        """Handle application closing"""
        if hasattr(self, 'detector'):
            self.detector.stop_camera()
        self.ui.on_closing()
        
    def run(self):
        """Start the application"""
        print("Starting application...")
        self.ui.run()

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.run()
