from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import csv
from tqdm import tqdm

model = AutoModelForImageClassification.from_pretrained("therealcyberlord/stanford-car-vit-patch16")
model.classifier = nn.Identity()

from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the size expected by the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # Normalize using the extractor's mean and std
        ])
        for folder in os.listdir(root):
            for image in os.listdir(os.path.join(root, folder)):
                self.images.append(os.path.join(root, folder, image))
                self.labels.append(folder)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


dataset = ImageDataset(r'D:\AllDatasets\ChuckTesla\Resized')
dataset = DataLoader(dataset, batch_size=8, shuffle=False)



os.makedirs('embeddings', exist_ok=True)

for img, label in tqdm(dataset):
    with torch.no_grad():
        
        out = model(img)['logits']

        embeddings = out.cpu().numpy()
        labels = label

        for i, label in enumerate(labels):
            embedding = embeddings[i]
            print(f"Embedding shape: {embedding.shape}")  # Debug print
            # Ensure embedding is 2D
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]
            file_path = os.path.join('embeddings', f'{label}.csv')
            # Write embeddings to CSV file
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(embedding)

            print(f"Embedding saved to: {file_path}")
