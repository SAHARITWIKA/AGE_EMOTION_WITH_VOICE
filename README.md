# Age, Emotion, and Gender Detection from Voice and Audio

This project detects **age**, **emotion**, and **gender** from voice recordings using pre-trained deep learning models. It processes audio input, extracts features, and runs predictions through CNN models.



## 📁 Project Structure

NULL_CLASS_TASK2_AGE_EMOTION_WITH_VOICE-MAIN/
│
├── data/ # Contains dataset organized by gender
│ ├── female/ # Audio files for female voices
│ └── male/ # Audio files for male voices
│
├── gender_model/
│ ├── init.py
│ ├── gender_cnn.py # CNN model for gender classification
│ └── train_gender.py # Training script for gender model
│
├── gui/
│ └── main_app.py # GUI-based app (possibly using Tkinter or PyQt)
│
├── models/
│ ├── age_model.pth # Trained PyTorch model for age prediction
│ ├── emotion_model.pth # Trained PyTorch model for emotion prediction
│ └── gender_cnn.pth # Trained gender model
│
├── notebooks/
│ ├── train_age_emotion_model.ipynb # Notebook to train age & emotion models
│ └── train_gender_model.ipynb # Notebook to train gender model
│
├── utils/
│ └── audio_preprocessing.py # Audio preprocessing functions
│
├── extract_gender_data.py # Gender dataset extractor
├── Book1_deduplicated.xlsx # Source data reference or metadata
├── tsv_test.py # Test script for tsv data
├── LICENSE
├── .gitignore
└── requirements.txt




---

## 📊 Dataset

The dataset is provided inside the `data/` directory:

- `data/male/` – Contains audio files for male speakers.
- `data/female/` – Contains audio files for female speakers.

Each audio file is used for training and testing the gender classification, and may also contribute to age and emotion classification if labeled accordingly.

---


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate



Install dependencies

pip install -r requirements.txt

Ensure trained model files are present in models/ folder.




_________________

1. Run the GUI Application

python gui/main_app.py

2. Train a Model
Train Gender Model:

Notebook: notebooks/train_gender_model.ipynb

Or Script: python gender_model/train_gender.py

Train Age & Emotion Models:

Notebook: notebooks/train_age_emotion_model.ipynb

3. Preprocess Audio
Use utils/audio_preprocessing.py for extracting and normalizing audio features before training or inference.

_____________________


Models Used
Gender Classification: Custom CNN (gender_cnn.py)

Age Prediction: Pretrained .pth model

Emotion Detection: Pretrained .pth model

📦 Dependencies
Ensure all packages in requirements.txt are installed. Common ones include:

torch

librosa

numpy

pandas

scikit-learn

matplotlib

openpyxl

tkinter or PyQt (for GUI)

