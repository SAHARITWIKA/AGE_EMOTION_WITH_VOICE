import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

from pydub.utils import which
print("FFmpeg path seen by pydub:", which("ffmpeg"))

# --- CONFIGURATION ---
tsv_path = "D:/AGE_EMOTION_WITH_VOICE/cv-corpus-22.0-delta-2025-06-20/en/validated.tsv"  # or validated.tsv
clips_dir = "D:/AGE_EMOTION_WITH_VOICE/cv-corpus-22.0-delta-2025-06-20/en/clips"
output_dir = "data"  # limit per gender to keep it lightweight

# --- Load TSV ---
df = pd.read_csv(tsv_path, sep='\t')

df['gender'] = df['gender'].map({
    'male_masculine': 'male',
    'female_feminine': 'female'
})

# Filter by gender
df = df[df['gender'].isin(['male', 'female'])]
df = df[df['path'].notna()]

# Limit to N samples per gender
male_df = df[df['gender'] == 'male']
female_df = df[df['gender'] == 'female']

min_len = min(len(male_df), len(female_df))
male_df = male_df.sample(n=min_len, random_state=42)
female_df = female_df.sample(n=min_len, random_state=42)


# Prepare output folders
os.makedirs(os.path.join(output_dir, "male"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "female"), exist_ok=True)

# --- Conversion Function ---
def convert_and_save(row, target_folder, count):
    mp3_path = os.path.join(clips_dir, row['path'])
    wav_path = os.path.join(output_dir, target_folder, f"{count}.wav")

    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound = sound.set_channels(1).set_frame_rate(22050)
        sound.export(wav_path, format="wav")
    except Exception as e:
        print(f"Failed on {mp3_path}: {e}")

# --- Process Male ---
print("Processing male voices...")
for i, row in tqdm(male_df.iterrows(), total=len(male_df)):
    convert_and_save(row, "male", i + 1)

# --- Process Female ---
print("Processing female voices...")
for i, row in tqdm(female_df.iterrows(), total=len(female_df)):
    convert_and_save(row, "female", i + 1)

print("✅ Gender dataset created in:", output_dir)
