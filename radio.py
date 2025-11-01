import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import sounddevice as sd
import os

# --- AUDIO EFFECT FUNCTIONS (Unchanged) ---
# ... (all our functions: bandpass_filter, add_distortion, add_fading, add_noise, add_crackles) ...
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def add_distortion(signal, gain=10):
    distorted_signal = np.tanh(signal * gain)
    return distorted_signal

def add_fading(signal, fs, fade_hz=0.5, min_volume=0.2):
    num_samples = len(signal)
    t = np.linspace(0., num_samples / fs, num_samples, endpoint=False)
    lfo = np.sin(2. * np.pi * fade_hz * t)
    mod_wave = lfo + 1.0
    mod_wave = mod_wave * (0.5 * (1.0 - min_volume))
    mod_wave = mod_wave + min_volume
    faded_signal = signal * mod_wave
    return faded_signal

def add_noise(signal, noise_amplitude=0.05):
    noise = np.random.normal(0, noise_amplitude, len(signal))
    signal_with_noise = signal + noise
    return signal_with_noise

def add_crackles(signal, fs, probability=0.0005, amplitude=0.25, min_ms=5, max_ms=75):
    output_signal = signal.copy()
    num_samples = len(signal)
    i = 0
    while i < num_samples:
        if np.random.rand() < probability:
            duration_ms = np.random.uniform(min_ms, max_ms)
            duration_samples = int((duration_ms / 1000.0) * fs)
            end_index = min(i + duration_samples, num_samples)
            actual_duration = end_index - i
            burst_noise = np.random.uniform(-amplitude, amplitude, actual_duration)
            output_signal[i:end_index] += burst_noise
            i += actual_duration
        else:
            i += 1
    return output_signal


# --- AUDIO PROCESSING LOGIC (Unchanged) ---
def process_audio(input_file, output_file, params, status_label, start_button):
    """
    This function runs the audio processing. It's designed to be run in a thread.
    """
    try:
        status_label.config(text="Status: Loading file...")
        fs, signal = wavfile.read(input_file)
        
        if signal.ndim == 2: signal = signal.mean(axis=1)
        original_dtype = signal.dtype
        signal = signal.astype(np.float32) / np.max(np.abs(signal))

        status_label.config(text="Status: Applying filter...")
        filtered = bandpass_filter(signal, params["low_cut"], params["high_cut"], fs)
        status_label.config(text="Status: Adding distortion...")
        distorted = add_distortion(filtered, params["distort_gain"])
        status_label.config(text="Status: Adding fading...")
        faded = add_fading(distorted, fs, params["fade_hz"], params["fade_min_vol"])
        status_label.config(text="Status: Adding hiss...")
        hiss = add_noise(faded, params["noise_amp"])
        status_label.config(text="Status: Adding crackles...")
        final_signal = add_crackles(
            hiss, fs, params["crackle_prob"], params["crackle_amp"],
            params["crackle_min_ms"], params["crackle_max_ms"]
        )
        
        status_label.config(text="Status: Normalizing...")
        final_signal = final_signal / np.max(np.abs(final_signal))
        if original_dtype == np.int16:
            final_signal = (final_signal * 32767).astype(np.int16)
        elif original_dtype == np.int32:
            final_signal = (final_signal * 2147483647).astype(np.int32)
        
        status_label.config(text="Status: Saving file...")
        wavfile.write(output_file, fs, final_signal)
        status_label.config(text=f"Status: Done! Saved to {output_file}")
    except Exception as e:
        status_label.config(text=f"Error: {e}")
    
    start_button.config(state=tk.NORMAL)


# --- GUI APPLICATION CLASS ---

class RadioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radio Effects (by Aladdstein)")

        # --- Recording Settings ---
        self.SAMPLE_RATE = 16000
        self.TEMP_FILE = "temp_recording.wav"
        self.is_recording = False
        self.recorded_frames = []

        # --- Define Presets ---
        self.PRESETS = {
            "Default": {
                "low_cut": "300.0", "high_cut": "3000.0", "distort_gain": "15.0",
                "fade_hz": "0.4", "fade_min_vol": "0.2", "noise_amp": "0.03",
                "crackle_prob": "0.0005", "crackle_amp": "0.25",
                "crackle_min_ms": "10.0", "crackle_max_ms": "100.0"
            },
            # ... (other presets are unchanged) ...
            "Walkie-Talkie": {
                "low_cut": "500.0", "high_cut": "2800.0", "distort_gain": "25.0",
                "fade_hz": "0.0", "fade_min_vol": "1.0", "noise_amp": "0.01",
                "crackle_prob": "0.0001", "crackle_amp": "0.1",
                "crackle_min_ms": "5.0", "crackle_max_ms": "20.0"
            },
            "Distant AM Station": {
                "low_cut": "200.0", "high_cut": "3500.0", "distort_gain": "5.0",
                "fade_hz": "0.2", "fade_min_vol": "0.1", "noise_amp": "0.15",
                "crackle_prob": "0.0002", "crackle_amp": "0.05",
                "crackle_min_ms": "50.0", "crackle_max_ms": "150.0"
            },
            "Numbers Station": {
                "low_cut": "400.0", "high_cut": "3200.0", "distort_gain": "3.0",
                "fade_hz": "0.1", "fade_min_vol": "0.5", "noise_amp": "0.10",
                "crackle_prob": "0.00005", "crackle_amp": "0.02",
                "crackle_min_ms": "10.0", "crackle_max_ms": "30.0"
            }
        }
        
        # --- Create Main Frame ---
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Create Widgets (Corrected Layout) ---
        self.create_io_widgets()      # <--- THIS IS NOW FIXED
        self.create_record_widgets() 
        self.create_preset_widgets() 
        self.create_param_widgets() 
        self.create_control_widgets()

        # Load default values on startup
        self.apply_preset("Default")

    def create_io_widgets(self):
        # <--- THIS FUNCTION IS NOW FIXED ---
        io_frame = ttk.Labelframe(self.main_frame, text="File Input / Output", padding="10")
        io_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Input File
        self.input_path_var = tk.StringVar()
        ttk.Label(io_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(io_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(io_frame, text="Browse", command=self.select_input_file).grid(row=0, column=2, padx=5)
        
        # Output File (This was missing)
        self.output_path_var = tk.StringVar()
        ttk.Label(io_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(io_frame, textvariable=self.output_path_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(io_frame, text="Browse", command=self.select_output_file).grid(row=1, column=2, padx=5)
        
        io_frame.columnconfigure(1, weight=1)

    def create_record_widgets(self): 
        rec_frame = ttk.Labelframe(self.main_frame, text="Option 2: Record from Mic", padding="10")
        rec_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.record_button = ttk.Button(rec_frame, text="Record Audio", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=5)
        
        self.record_status_label = ttk.Label(rec_frame, text="Status: Ready to record")
        self.record_status_label.grid(row=0, column=1, padx=10, sticky=tk.W)
        
        rec_frame.columnconfigure(1, weight=1)

    def create_preset_widgets(self):
        preset_frame = ttk.Labelframe(self.main_frame, text="Load Preset", padding="10")
        preset_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        col = 0
        for preset_name in self.PRESETS:
            btn = ttk.Button(preset_frame, text=preset_name, 
                             command=lambda p=preset_name: self.apply_preset(p))
            btn.grid(row=0, column=col, padx=5, pady=5)
            col += 1

    def create_param_widgets(self):
        param_frame = ttk.Labelframe(self.main_frame, text="Effect Parameters", padding="10")
        param_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        self.param_vars = {
            "low_cut": tk.StringVar(), "high_cut": tk.StringVar(),
            "distort_gain": tk.StringVar(), "fade_hz": tk.StringVar(),
            "fade_min_vol": tk.StringVar(), "noise_amp": tk.StringVar(),
            "crackle_prob": tk.StringVar(), "crackle_amp": tk.StringVar(),
            "crackle_min_ms": tk.StringVar(), "crackle_max_ms": tk.StringVar()
        }
        param_labels = {
            "low_cut": "Low Cut (Hz):", "high_cut": "High Cut (Hz):",
            "distort_gain": "Distortion Gain:", "fade_hz": "Fade Speed (Hz):",
            "fade_min_vol": "Fade Min Vol (0-1):", "noise_amp": "Hiss Amplitude:",
            "crackle_prob": "Crackle Probability:", "crackle_amp": "Crackle Amplitude:",
            "crackle_min_ms": "Crackle Min (ms):", "crackle_max_ms": "Crackle Max (ms):"
        }
        row_num, col_num = 0, 0
        for key, label_text in param_labels.items():
            ttk.Label(param_frame, text=label_text).grid(row=row_num, column=col_num, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(param_frame, textvariable=self.param_vars[key], width=10).grid(row=row_num, column=col_num+1, sticky=tk.W, padx=5, pady=2)
            col_num += 2
            if col_num > 2: col_num, row_num = 0, row_num + 1
                
    def create_control_widgets(self):
        bottom_frame = ttk.Frame(self.main_frame)
        bottom_frame.grid(row=4, column=0, sticky=(tk.E, tk.W), pady=10)

        self.start_button = ttk.Button(bottom_frame, text="Start Conversion", command=self.start_conversion_thread)
        self.start_button.pack(side=tk.TOP, pady=5)
        
        self.status_label = ttk.Label(bottom_frame, text="Status: Ready")
        self.status_label.pack(side=tk.TOP, pady=5)
        
        bottom_frame.columnconfigure(0, weight=1)

    # --- GUI Helper Functions ---

    def select_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.input_path_var.set(file_path)
            
    def select_output_file(self):
        # <--- THIS FUNCTION IS USED BY THE NEW BUTTON ---
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.output_path_var.set(file_path)

    def apply_preset(self, preset_name):
        try:
            preset_values = self.PRESETS[preset_name]
            for key, value in preset_values.items():
                if key in self.param_vars:
                    self.param_vars[key].set(value)
            self.status_label.config(text=f"Status: Loaded '{preset_name}' preset.")
        except KeyError:
            self.status_label.config(text=f"Error: Preset '{preset_name}' not found.")

    # --- Recording Functions ---

    def toggle_recording(self):
        if self.is_recording:
            # --- STOP RECORDING ---
            self.is_recording = False
            self.record_button.config(text="Processing...", state=tk.DISABLED)
            self.record_status_label.config(text="Status: Stopping and saving...")
        else:
            # --- START RECORDING ---
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.record_status_label.config(text="Status: Recording...")
            self.recorded_frames = [] 
            
            threading.Thread(target=self.record_audio, daemon=True).start()

    def record_audio(self):
        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, 
                                channels=1, 
                                dtype='int16') as stream:
                while self.is_recording:
                    data, overflow = stream.read(1024)
                    self.recorded_frames.append(data)
            
            self.record_status_label.config(text="Status: Saving temporary file...")
            recording = np.concatenate(self.recorded_frames, axis=0)
            wavfile.write(self.TEMP_FILE, self.SAMPLE_RATE, recording)
            temp_file_path = os.path.abspath(self.TEMP_FILE)
            
            self.input_path_var.set(temp_file_path)
            self.record_status_label.config(text="Status: Ready to record")
            self.status_label.config(text=f"Recording saved. Ready to convert.")
        except Exception as e:
            self.record_status_label.config(text=f"Error: {e}")
        
        self.record_button.config(text="Record Audio", state=tk.NORMAL)

    # --- Conversion Function ---

    def start_conversion_thread(self):
        input_file = self.input_path_var.get()
        # <--- THIS VARIABLE NOW WORKS AGAIN ---
        output_file = self.output_path_var.get()
        
        if not input_file or not output_file:
            self.status_label.config(text="Status: Please select input AND output files.")
            return
        if not os.path.exists(input_file):
            self.status_label.config(text=f"Error: Input file not found: {input_file}")
            return

        try:
            params = {key: float(var.get()) for key, var in self.param_vars.items()}
        except ValueError:
            self.status_label.config(text="Error: Invalid parameter. Please enter numbers only.")
            return
            
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Starting conversion...")
        
        threading.Thread(
            target=process_audio, 
            args=(input_file, output_file, params, self.status_label, self.start_button),
            daemon=True
        ).start()

# --- Main execution ---
def main():
    root = tk.Tk()
    app = RadioApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
