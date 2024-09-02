import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from transformers import AutoModelForImageClassification

# Load the model
model = AutoModelForImageClassification.from_pretrained("therealcyberlord/stanford-car-vit-patch16")
model.classifier = nn.Identity()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize using the extractor's mean and std
])

#img = Image.open(r'cybertruck.png').convert("RGB")
img = transform(img).unsqueeze(0)

# Get the image embedding
with torch.no_grad():
    image_embedding = model(img)['logits'].cpu().numpy()

# Ensure image_embedding is 2D
if image_embedding.ndim == 1:
    image_embedding = image_embedding[np.newaxis, :]

def load_embeddings(file_path):
    embeddings = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            embeddings.append(np.array(row, dtype=float))
    return np.array(embeddings)

def compare_image_to_classes(image_embedding, class_files, threshold):
    class_scores = []
    for class_file in class_files:
        class_embedding = load_embeddings(class_file)
        # Ensure class_embedding is 2D
        if class_embedding.ndim == 1:
            class_embedding = class_embedding[np.newaxis, :]
        
        #class_embedding = class_embedding.mean(axis=0)[None, :]
        sim = cosine_similarity(image_embedding, class_embedding)
        # Count the number of similarities above the threshold
        print(sim)
        score = np.sum(sim > threshold)
        normalized_score = score / len(class_embedding)
        print(normalized_score)
        class_scores.append(normalized_score)
    
    return np.array(class_scores)

def get_most_similar_class(image_embedding, class_files, threshold=0.6):
    scores = compare_image_to_classes(image_embedding, class_files, threshold)
    most_similar_class_index = np.argmax(scores)
    most_similar_class = class_files[most_similar_class_index]
    return most_similar_class, scores[most_similar_class_index]

# Example usage
class_files = ['embeddings/3.csv', 'embeddings/Cybertruck.csv', 'embeddings/non_teslas.csv', 'embeddings/S.csv', 'embeddings/X.csv', 'embeddings/Y.csv']
most_similar_class, score = get_most_similar_class(image_embedding, class_files)
print(f"The most similar class is {most_similar_class} with a score of {score}")






