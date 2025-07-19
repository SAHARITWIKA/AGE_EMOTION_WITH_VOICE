import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import numpy as np
import torch
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "D:\AGE_EMOTION_WITH_VOICE")))
from gender_model.gender_cnn import GenderCNN

# ========== GUI Setup ========== #
app = tb.Window(themename="darkly")
app.title("Voice Age & Emotion Detector")
app.geometry("800x550")
app.resizable(False, False)

# Optional: background image (ensure you have dark_bg.jpg)
if os.path.exists("dark_bg.jpg"):
    from tkinter import Label
    from PIL import Image, ImageTk
    bg_img = Image.open("dark_bg.jpg").resize((800, 550))
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = Label(app, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# ========== UI Components ========== #

header = tb.Label(app, text="🔊 Voice-Based Age & Emotion Detection", font=("Helvetica", 20, "bold"), bootstyle="info")
header.pack(pady=30)

upload_btn = tb.Button(app, text="🎤 Upload Audio File", bootstyle="success outline", width=30)
upload_btn.pack(pady=10)

file_label = tb.Label(app, text="", font=("Helvetica", 12), bootstyle="secondary")
file_label.pack(pady=5)

result_label = tb.Label(app, text="Result will appear here...", font=("Helvetica", 14), bootstyle="warning")
result_label.pack(pady=20)

progress = tb.Progressbar(app, mode='indeterminate', bootstyle="info-striped", length=300)
progress.pack(pady=10)
progress.stop()  # Initially not running


# ========== Waveform Display ========== #

def show_waveform(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.set_facecolor("#1e1e1e")
    ax.plot(audio, color="#4CAF50")
    ax.set_title("Waveform", color="white")
    ax.set_xlabel("Time", color="gray")
    ax.set_ylabel("Amplitude", color="gray")
    ax.tick_params(colors="gray")

    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.5, rely=0.8, anchor="center")

# ========= Preprocess_logic ============ #
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.fix_length(y, size=22050 * 5)  # 5 seconds max
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())  # normalize
    mel_db = np.resize(mel_db, (128, 128))  # reshape to fixed size
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()  # shape (1, 1, 128, 128)
    return mel_tensor

# ========= Gender_model_upload ========= #

gender_model = GenderCNN()
gender_model.load_state_dict(torch.load("models/gender_cnn.pth", map_location=torch.device('cpu')))
gender_model.eval()

# ========== File Upload Logic ========== #

def upload_file():
    file_path = filedialog.askopenfilename(
        title="Choose an audio file",
        filetypes=[("Audio Files", "*.wav *.mp3")]
    )
    if file_path:
        filename = os.path.basename(file_path)
        file_label.config(text=f"🎧 {filename} selected", bootstyle="success")
        result_label.config(text="Analyzing gender...", bootstyle="info")

        # Start animated loading spinner
        progress.start()
        app.update_idletasks()

        # Show waveform
        show_waveform(file_path)

        # Preprocess audio → tensor
        input_tensor = preprocess_audio(file_path)

        # Predict gender using CNN
        with torch.no_grad():
            gender_output = gender_model(input_tensor).item()

        # Stop progress bar
        progress.stop()

        # Display result
        if gender_output < 0.5:
            result_label.config(
                text="🚫 Female voice detected. Please upload a male voice.",
                bootstyle="danger"
            )
        else:
            result_label.config(
                text="✅ Male voice detected. Proceeding to age prediction...",
                bootstyle="success"
            )
            # 👉 You'll call age prediction here later


upload_btn.config(command=upload_file)

# ========== Run App ========== #

app.mainloop()
