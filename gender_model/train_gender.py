# train_gender.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from gender_cnn import GenderCNN
from tqdm import tqdm

class GenderVoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, gender in enumerate(["female", "male"]):  # 0=female, 1=male
            gender_dir = os.path.join(root_dir, gender)
            for fname in os.listdir(gender_dir):
                if fname.endswith(".wav"):
                    self.samples.append((os.path.join(gender_dir, fname), label))
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.util.fix_length(y, size=22050 * 5)  # 5 seconds
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())  # normalize
        mel_db = np.resize(mel_db, (128, 128))  # standard shape
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()  # (1, 128, 128)
        return mel_tensor, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderCNN().to(device)
dataset = GenderVoiceDataset("data")
# After dataset creation
print("Total samples:", len(dataset))
male = sum(1 for x in dataset.samples if x[1] == 1)
female = sum(1 for x in dataset.samples if x[1] == 0)
print(f"Male: {male} | Female: {female}")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} Loss: {running_loss:.4f} Accuracy: {acc:.2%}")
    print(f"Predictions: {predictions[:10].squeeze().tolist()}")
    print(f"Labels: {labels[:10].squeeze().tolist()}")

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gender_cnn.pth")
print("✅ Model saved to models/gender_cnn.pth")
