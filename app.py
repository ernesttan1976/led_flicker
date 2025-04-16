import cv2
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tempfile
import os
import threading
import traceback
import sys

class LEDFlickerDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("LED Flicker Frequency Stability Detector")
        self.root.geometry("900x800")
        
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = None
        self.result_image = None
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        # Title
        title_label = tk.Label(self.root, text="LED Flicker Frequency Stability Detector", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.root, text=(
            "1. Click 'Browse' to select a video containing LED lights\n"
            "2. Adjust the brightness threshold if needed\n"
            "3. Click 'Analyze' to detect LEDs with unstable flicker patterns\n"
            "4. LEDs with unstable frequencies will be circled in red"
        ), justify=tk.LEFT)
        instructions.pack(pady=10)
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.file_label = tk.Label(file_frame, text="No file selected", width=50, anchor="w")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        browse_button = tk.Button(file_frame, text="Browse", command=self.safe_browse_file)
        browse_button.pack(side=tk.RIGHT, padx=5)
        
        # Threshold slider
        threshold_frame = tk.Frame(self.root)
        threshold_frame.pack(fill=tk.X, padx=20, pady=10)
        
        threshold_label = tk.Label(threshold_frame, text="LED Brightness Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.IntVar(value=200)
        threshold_slider = ttk.Scale(threshold_frame, from_=50, to=250, orient=tk.HORIZONTAL, 
                                    variable=self.threshold_var, length=300)
        threshold_slider.pack(side=tk.LEFT, padx=5)
        
        threshold_value = tk.Label(threshold_frame, textvariable=self.threshold_var)
        threshold_value.pack(side=tk.LEFT, padx=5)
        
        # Stability threshold frame
        stability_frame = tk.Frame(self.root)
        stability_frame.pack(fill=tk.X, padx=20, pady=10)
        
        stability_label = tk.Label(stability_frame, text="Frequency Stability Threshold (%):")
        stability_label.pack(side=tk.LEFT, padx=5)
        
        self.stability_var = tk.DoubleVar(value=10.0)
        stability_slider = ttk.Scale(stability_frame, from_=1.0, to=50.0, orient=tk.HORIZONTAL, 
                                    variable=self.stability_var, length=300)
        stability_slider.pack(side=tk.LEFT, padx=5)
        
        stability_value = tk.Label(stability_frame, textvariable=self.stability_var)
        stability_value.pack(side=tk.LEFT, padx=5)
        
        # Time window size frame
        window_frame = tk.Frame(self.root)
        window_frame.pack(fill=tk.X, padx=20, pady=10)
        
        window_label = tk.Label(window_frame, text="Analysis Window Size (frames):")
        window_label.pack(side=tk.LEFT, padx=5)
        
        self.window_var = tk.IntVar(value=60)
        window_slider = ttk.Scale(window_frame, from_=20, to=120, orient=tk.HORIZONTAL, 
                                 variable=self.window_var, length=300)
        window_slider.pack(side=tk.LEFT, padx=5)
        
        window_value = tk.Label(window_frame, textvariable=self.window_var)
        window_value.pack(side=tk.LEFT, padx=5)
        
        # Analysis frames slider
        frames_frame = tk.Frame(self.root)
        frames_frame.pack(fill=tk.X, padx=20, pady=10)
        
        frames_label = tk.Label(frames_frame, text="Total Frames to Analyze:")
        frames_label.pack(side=tk.LEFT, padx=5)
        
        self.frames_var = tk.IntVar(value=300)
        frames_slider = ttk.Scale(frames_frame, from_=100, to=1000, orient=tk.HORIZONTAL, 
                                 variable=self.frames_var, length=300)
        frames_slider.pack(side=tk.LEFT, padx=5)
        
        frames_value = tk.Label(frames_frame, textvariable=self.frames_var)
        frames_value.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        analyze_button = tk.Button(self.root, text="Analyze Video", command=self.safe_analyze_video)
        analyze_button.pack(pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Select a video file to begin")
        status_label = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 10, "italic"))
        status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.pack(pady=10)
        
        # Image display
        self.image_frame = tk.Frame(self.root, width=800, height=500, bg="gray")
        self.image_frame.pack(pady=10)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
    
    def safe_browse_file(self):
        """Wrapper for browse_file with exception handling"""
        try:
            self.browse_file()
        except Exception as e:
            error_msg = f"Error during file browsing: {str(e)}\n{traceback.format_exc()}"
            self.status_var.set(error_msg)
            print(error_msg, file=sys.stderr)
    
    def browse_file(self):
        """Open file dialog to select a video file"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*")
        ]
        
        # Use the simplest form of the file dialog for maximum compatibility
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=file_types
        )
        
        # Only proceed if a file was actually selected
        if file_path and len(file_path) > 0:
            self.video_path = file_path
            # Truncate the filename if it's too long
            basename = os.path.basename(file_path)
            if len(basename) > 40:
                basename = basename[:37] + "..."
            self.file_label.config(text=basename)
            self.status_var.set("Video selected. Click 'Analyze Video' to begin processing")
    
    def safe_analyze_video(self):
        """Wrapper for analyze_video with exception handling"""
        try:
            self.analyze_video()
        except Exception as e:
            error_msg = f"Error during video analysis: {str(e)}\n{traceback.format_exc()}"
            self.status_var.set(f"Error: {str(e)}")
            print(error_msg, file=sys.stderr)
            # Stop progress bar if it was running
            self.progress.stop()
            # Re-enable analyze button
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Button) and widget["text"] == "Analyze Video":
                    widget.config(state=tk.NORMAL)
    
    def analyze_video(self):
        """Start video analysis process"""
        if not self.video_path:
            self.status_var.set("Please select a video file first")
            return
        
        # Start processing in a separate thread to keep UI responsive
        self.progress.start()
        self.status_var.set("Processing video... This may take a moment")
        
        # Disable analyze button during processing
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button) and widget["text"] == "Analyze Video":
                widget.config(state=tk.DISABLED)
        
        # Start processing thread
        threading.Thread(target=self._process_video, daemon=True).start()
    
    def _process_video(self):
        """Process video in a separate thread"""
        threshold = self.threshold_var.get()
        frames_to_analyze = self.frames_var.get()
        stability_threshold = self.stability_var.get() / 100.0  # Convert from percentage to decimal
        window_size = self.window_var.get()
        
        try:
            # Process the video
            message, result_image = self.detect_frequency_stability(
                self.video_path, 
                threshold, 
                frames_to_analyze,
                window_size,
                stability_threshold
            )
            
            # Update UI in the main thread
            self.root.after(0, lambda: self._update_ui(message, result_image))
            
        except Exception as e:
            error_msg = f"Error during video processing: {str(e)}\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)
            # Update UI with error in the main thread
            self.root.after(0, lambda: self._update_ui(f"Error: {str(e)}", None))
    
    def _update_ui(self, message, result_image):
        """Update UI with processing results"""
        # Stop progress bar
        self.progress.stop()
        
        # Update status
        self.status_var.set(message)
        
        # Re-enable analyze button
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button) and widget["text"] == "Analyze Video":
                widget.config(state=tk.NORMAL)
        
        # Display result image if available
        if result_image is not None:
            try:
                # Convert OpenCV image to PIL format
                pil_image = Image.fromarray(result_image)
                
                # Resize to fit display area if needed
                img_width, img_height = pil_image.size
                max_width = 800
                max_height = 500
                
                # Calculate resize ratio while preserving aspect ratio
                if img_width > max_width or img_height > max_height:
                    ratio = min(max_width / img_width, max_height / img_height)
                    new_width = int(img_width * ratio)
                    new_height = int(img_height * ratio)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to Tkinter format
                self.tk_image = ImageTk.PhotoImage(pil_image)
                
                # Update image display
                self.image_label.config(image=self.tk_image)
                self.image_label.image = self.tk_image  # Keep a reference to prevent garbage collection
            except Exception as e:
                error_msg = f"Error displaying image: {str(e)}\n{traceback.format_exc()}"
                print(error_msg, file=sys.stderr)
                self.status_var.set(f"Error displaying image: {str(e)}")
    
    def detect_dominant_frequency(self, signal, fps):
        """
        Detect the dominant frequency in a signal using FFT
        
        Args:
            signal: Time series signal data
            fps: Frames per second of the video
            
        Returns:
            frequency: Dominant frequency in Hz
            amplitude: Amplitude of the dominant frequency
        """
        # Normalize signal
        normalized = signal - np.mean(signal)
        
        # Apply smoothing to reduce noise
        window_size = 3
        if len(normalized) > window_size:
            smoothed = np.convolve(normalized, np.ones(window_size)/window_size, mode='valid')
        else:
            smoothed = normalized
        
        # Apply FFT
        n = len(smoothed)
        if n <= 1:
            return 0, 0
            
        # Compute FFT
        fft_result = np.abs(fft(smoothed))
        # Use only first half (positive frequencies)
        fft_result = fft_result[1:n//2]
        
        # Calculate frequency axis
        freqs = np.fft.fftfreq(n, d=1/fps)[1:n//2]
        
        # Find peaks in frequency domain
        min_height = max(10, np.max(fft_result) * 0.3)
        peaks, _ = find_peaks(fft_result, height=min_height, distance=3)
        
        if len(peaks) > 0:
            # Sort peaks by amplitude
            sorted_peaks = sorted([(freqs[i], fft_result[i]) for i in peaks], 
                                key=lambda x: x[1], reverse=True)
            
            # Get the dominant frequency and its amplitude
            dominant_freq, amplitude = sorted_peaks[0]
            return dominant_freq, amplitude
        else:
            return 0, 0
    
    def detect_frequency_stability(self, video_path, threshold=200, max_frames=300, 
                                 window_size=60, stability_threshold=0.1):
        """
        Analyze a video to find LEDs with unstable flicker frequencies.
        
        Args:
            video_path: Path to the video file
            threshold: Brightness threshold for LED detection
            max_frames: Maximum number of frames to analyze
            window_size: Size of time windows for frequency analysis
            stability_threshold: Maximum allowed relative standard deviation of frequency
            
        Returns:
            message: Status message
            result_frame: Frame with detected LEDs marked
        """
        # Step 1: Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file", None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Step 2: Get first frame for initial detection
        ret, first_frame = cap.read()
        if not ret:
            return "Error: Could not read first frame", None
        
        # Keep a copy of the original frame for display
        display_frame = first_frame.copy()
        
        # Step 3: Detect LED positions (using simple thresholding)
        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # Apply threshold to find bright spots (LEDs)
        _, thresh = cv2.threshold(gray_first, threshold, 255, cv2.THRESH_BINARY)
        
        # Use morphological operations to clean up the thresholded image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Use connected components to find LED positions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        
        # Filter out the background and small regions
        led_positions = []
        led_rois = []  # Store the region of interest for each LED
        
        for i in range(1, num_labels):  # Skip background label 0
            if stats[i, cv2.CC_STAT_AREA] > 5:  # Minimum area for an LED
                x = int(centroids[i][0])
                y = int(centroids[i][1])
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate region of interest (ROI) with padding
                padding = 2
                x_min = max(0, int(x - width/2) - padding)
                y_min = max(0, int(y - height/2) - padding)
                x_max = min(gray_first.shape[1], int(x + width/2) + padding)
                y_max = min(gray_first.shape[0], int(y + height/2) + padding)
                
                led_positions.append((x, y))
                led_rois.append((x_min, y_min, x_max, y_max))
        
        if len(led_positions) == 0:
            return f"No LEDs detected. Try lowering the threshold (current: {threshold})", None
        
        # Step 4: Process frames to extract LED brightness over time
        frames_to_analyze = min(frame_count, max_frames)
        num_leds = len(led_positions)
        brightness_data = np.zeros((num_leds, frames_to_analyze))
        
        # Reset to beginning of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_idx in range(frames_to_analyze):
            ret, frame = cap.read()
            if not ret:
                brightness_data = brightness_data[:, :frame_idx]  # Truncate data if we run out of frames
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for led_idx, (_, _, _, _) in enumerate(led_rois):
                x_min, y_min, x_max, y_max = led_rois[led_idx]
                roi = gray[y_min:y_max, x_min:x_max]
                
                if roi.size > 0:
                    # Calculate mean brightness of the ROI
                    brightness_data[led_idx, frame_idx] = np.mean(roi)
        
        cap.release()
        
        # Step 5: Analyze frequency stability over time for each LED
        stability_data = []
        average_freq_data = []
        stability_metrics = []
        
        # Calculate how many windows we'll have
        num_windows = max(1, (frames_to_analyze - window_size) // (window_size // 2) + 1)
        
        for led_idx in range(num_leds):
            led_signal = brightness_data[led_idx]
            
            # Skip processing if signal is too weak or flat
            if np.std(led_signal) < 2.0:
                stability_data.append(0)
                average_freq_data.append(0)
                stability_metrics.append(0)
                continue
                
            # Analyze frequency in overlapping windows
            window_frequencies = []
            window_amplitudes = []
            
            for win_idx in range(num_windows):
                start_idx = win_idx * (window_size // 2)
                end_idx = min(start_idx + window_size, frames_to_analyze)
                
                # Skip if window is too small
                if end_idx - start_idx < window_size // 2:
                    continue
                    
                window_signal = led_signal[start_idx:end_idx]
                
                # Calculate dominant frequency in this window
                freq, amp = self.detect_dominant_frequency(window_signal, fps)
                
                # Only consider frequencies with significant amplitude
                if freq > 0.5 and amp > 5:
                    window_frequencies.append(freq)
                    window_amplitudes.append(amp)
            
            # Calculate stability metrics if we have enough data
            if len(window_frequencies) >= 3:
                avg_freq = np.mean(window_frequencies)
                
                # Calculate coefficient of variation (relative standard deviation)
                std_freq = np.std(window_frequencies)
                relative_std = std_freq / avg_freq if avg_freq > 0 else 0
                
                # Calculate mean absolute percentage deviation
                deviations = np.abs(np.array(window_frequencies) - avg_freq) / avg_freq
                mean_deviation = np.mean(deviations)
                
                # Store stability metrics
                stability_metric = max(relative_std, mean_deviation)
                stability_data.append(stability_metric)
                average_freq_data.append(avg_freq)
                stability_metrics.append(stability_metric)
            else:
                # Not enough data to calculate stability
                stability_data.append(0)
                average_freq_data.append(0)
                stability_metrics.append(0)
        
        # Step 6: Identify LEDs with unstable frequencies
        unstable_leds = []
        for i, stability in enumerate(stability_data):
            # Check if stability metric exceeds threshold and we have a valid frequency
            if stability > stability_threshold and average_freq_data[i] > 1.0:
                unstable_leds.append(i)
        
        # Step 7: Create output visualization
        result_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Display frequency information for each LED
        for i, (x, y) in enumerate(led_positions):
            avg_freq = average_freq_data[i]
            stability = stability_data[i]
            
            # Determine if LED is defective (unstable)
            is_unstable = i in unstable_leds
            
            if is_unstable:
                color = (255, 0, 0)  # Red for unstable
                variation = stability * 100  # Convert to percentage
                radius = 20  # Larger circle for unstable LEDs
            else:
                if avg_freq > 0:
                    color = (0, 255, 0)  # Green for stable
                    radius = 15
                else:
                    color = (200, 200, 200)  # Gray for LEDs with insufficient data
                    radius = 10
                    
            # Draw circle around LED
            cv2.circle(result_frame, (x, y), radius, color, 2)
            
            # Add frequency label if significant
            if avg_freq > 0.5:
                freq_text = f"{avg_freq:.1f} Hz"
                cv2.putText(result_frame, freq_text, (x-20, y-radius-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Mark unstable LEDs more prominently
            if is_unstable:
                cv2.circle(result_frame, (x, y), radius+10, color, 2)
                cv2.putText(result_frame, "UNSTABLE", (x-40, y+radius+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                variation_text = f"±{variation:.1f}% variation"
                cv2.putText(result_frame, variation_text, (x-60, y+radius+35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Status message
        if len(unstable_leds) > 0:
            message = f"Found {len(unstable_leds)} LEDs with unstable flicker frequencies"
        else:
            message = "All detected LEDs have stable flicker patterns"
        
        # Add summary to image
        cv2.putText(result_frame, f"Analyzed {frames_to_analyze} frames at {fps:.1f} FPS", 
                   (10, result_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(result_frame, f"Window size: {window_size} frames, Stability threshold: {stability_threshold*100:.1f}%", 
                   (10, result_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(result_frame, f"Detected {len(led_positions)} LEDs, {len(unstable_leds)} unstable", 
                   (10, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return message, result_frame

if __name__ == "__main__":
    root = tk.Tk()
    app = LEDFlickerDetector(root)
    root.mainloop()