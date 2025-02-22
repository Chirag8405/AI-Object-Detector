import customtkinter as ctk
from PIL import Image
import numpy as np
from customtkinter import CTkImage

class ApplicationUI:
    def __init__(self):
        """Initialize the main application window"""
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.window = ctk.CTk()
        self.window.title("AI Object Detection")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 600)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create left panel 
        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Create video frame
        self.video_frame = ctk.CTkFrame(self.left_panel)
        self.video_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create video label with initial text
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Camera Feed Not Started",
            font=("Arial", 16)
        )
        self.video_label.pack(fill="both", expand=True)
        
        # Create control panel under video
        self.control_panel = ctk.CTkFrame(self.left_panel)
        self.control_panel.pack(fill="x", padx=5, pady=5)
        
        # Add controls
        self.running = False
        self.control_button = ctk.CTkButton(
            self.control_panel,
            text="Start Detection",
            command=self.toggle_detection,
            width=150,
            height=40,
            font=("Arial", 14, "bold")
        )
        self.control_button.pack(side="left", padx=10, pady=10)
        
        # Add confidence threshold slider
        self.conf_label = ctk.CTkLabel(
            self.control_panel,
            text="Confidence Threshold:",
            font=("Arial", 12)
        )
        self.conf_label.pack(side="left", padx=(20, 5), pady=10)
        
        self.conf_slider = ctk.CTkSlider(
            self.control_panel,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            width=150
        )
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side="left", padx=5, pady=10)
        
        self.conf_value_label = ctk.CTkLabel(
            self.control_panel,
            text="0.50",
            font=("Arial", 12)
        )
        self.conf_value_label.pack(side="left", padx=5, pady=10)
        
        # Create right panel 
        self.right_panel = ctk.CTkFrame(self.main_frame, width=300)
        self.right_panel.pack(side="right", fill="y", padx=5, pady=5)
        self.right_panel.pack_propagate(False)
        
        # Add title to right panel
        self.title_label = ctk.CTkLabel(
            self.right_panel,
            text="Detected Objects",
            font=("Arial", 20, "bold")
        )
        self.title_label.pack(pady=10)
        
        # Add detection stats
        self.stats_frame = ctk.CTkFrame(self.right_panel)
        self.stats_frame.pack(fill="x", padx=10, pady=5)
        
        self.total_detections_label = ctk.CTkLabel(
            self.stats_frame,
            text="Total Detections: 0",
            font=("Arial", 14)
        )
        self.total_detections_label.pack(pady=5)
        
        # Create scrollable frame for detections
        self.detections_frame = ctk.CTkScrollableFrame(
            self.right_panel,
            label_text="Detection List",
            label_font=("Arial", 14, "bold")
        )
        self.detections_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bind confidence slider update
        self.conf_slider.configure(command=self._update_conf_label)
        
        # Store current image reference
        self.current_image = None
        
    def _update_conf_label(self, value):
        """Update confidence threshold label"""
        self.conf_value_label.configure(text=f"{float(value):.2f}")
        
    def toggle_detection(self):
        """Toggle between starting and stopping detection"""
        self.running = not self.running
        if self.running:
            self.control_button.configure(
                text="Stop Detection",
                fg_color="#c42b1c",  
                hover_color="#a62215"
            )
            self.video_label.configure(text="")  
        else:
            self.control_button.configure(
                text="Start Detection",
                fg_color=["#3a7ebf", "#1f538d"], 
                hover_color=["#325882", "#14375e"]
            )
            
    def update_frame(self, ctk_image):
        """Update the video frame with new image"""
        if ctk_image is not None:
            if not isinstance(ctk_image, ctk.CTkImage):
                if isinstance(ctk_image, Image.Image):  
                    ctk_image = ctk.CTkImage(light_image=ctk_image, size=(640, 480))
                else:
                    print(f"Invalid image type: {type(ctk_image)}")  
                    return 

            self.current_image = ctk_image
            self.video_label.configure(image=self.current_image, text="")
        else:
            self.video_label.configure(text="Camera Feed Not Started", image=None)

            
    def update_detections(self, detections):
        """Update the detections list in the sidebar"""
        self.total_detections_label.configure(text=f"Total Detections: {len(detections)}")
        
        for widget in self.detections_frame.winfo_children():
            widget.destroy()
            
        # Add new detections
        for i, det in enumerate(detections):
            det_frame = ctk.CTkFrame(self.detections_frame)
            det_frame.pack(fill="x", pady=2, padx=5)
            
            # Add detection info
            class_label = ctk.CTkLabel(
                det_frame,
                text=f"{det['class']}",
                font=("Arial", 13, "bold")
            )
            class_label.pack(side="left", padx=5, pady=2)
            
            conf_label = ctk.CTkLabel(
                det_frame,
                text=f"Conf: {det['confidence']:.2f}",
                font=("Arial", 12),
                text_color="gray70"
            )
            conf_label.pack(side="right", padx=5, pady=2)
            
    def get_confidence_threshold(self):
        """Get current confidence threshold value"""
        return self.conf_slider.get()
            
    def run(self):
        """Start the application main loop"""
        self.window.mainloop()
        
    def on_closing(self):
        """Handle application closing"""
        self.window.quit()
        self.window.destroy()
