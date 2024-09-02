import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import os
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from transformers import AutoModelForImageClassification
import matplotlib.pyplot as plt

# Load the object detection model
detection_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
detection_model.eval()

# Load the classification model
classification_model = AutoModelForImageClassification.from_pretrained("therealcyberlord/stanford-car-vit-patch16")
classification_model.classifier = nn.Identity()
classification_model.eval()

# Define the transforms
detection_transform = transforms.Compose([transforms.ToTensor()])
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_car_crop(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = detection_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = detection_model(image_tensor)[0]
    
    # Filter for car detections (class 3 in COCO dataset)
    car_detections = prediction['boxes'][prediction['labels'] == 3]
    car_scores = prediction['scores'][prediction['labels'] == 3]
    
    if len(car_detections) == 0:
        print("No cars detected in the image.")
        return None
    
    # Get the detection with the highest score
    best_detection = car_detections[car_scores.argmax()]
    x1, y1, x2, y2 = map(int, best_detection.tolist())  # Convert tensor to list of ints
    
    # Crop the image
    car_crop = image.crop((x1, y1, x2, y2))
    return car_crop

def get_image_embedding(image):
    img = classification_transform(image).unsqueeze(0)
    with torch.no_grad():
        image_embedding = classification_model(img)['logits'].cpu().numpy()
    return image_embedding

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
        if class_embedding.ndim == 1:
            class_embedding = class_embedding[np.newaxis, :]
        
        sim = cosine_similarity(image_embedding, class_embedding)
        score = np.sum(sim > threshold)
        normalized_score = score / len(class_embedding)
        print(f"Score for {class_file}: {normalized_score}")
        class_scores.append(normalized_score)
    
    return np.array(class_scores)

def get_most_similar_class(image_embedding, class_files, threshold=0.6):
    scores = compare_image_to_classes(image_embedding, class_files, threshold)
    most_similar_class_index = np.argmax(scores)
    most_similar_class = class_files[most_similar_class_index]
    return most_similar_class, scores[most_similar_class_index]

# Main execution
image_path = r'Resized\non_teslas\07_44_18_image.png'
#image_path = r'Resized\S\08_24_01_image.png'
#image_path = r'Resized\Y\18_07_09_image.png'
#image_path = r'Resized\Cybertruck\08_29_00_image.png'
#image_path = r'Resized\3\17_29_02_image.png'

car_crop = get_car_crop(image_path)
plt.imshow(car_crop)
plt.show()

if car_crop is not None:
    image_embedding = get_image_embedding(car_crop)
    
    # Ensure image_embedding is 2D
    if image_embedding.ndim == 1:
        image_embedding = image_embedding[np.newaxis, :]
    
    class_files = ['embeddings/3.csv', 'embeddings/Cybertruck.csv', 'embeddings/non_teslas.csv', 'embeddings/S.csv', 'embeddings/X.csv', 'embeddings/Y.csv']
    most_similar_class, score = get_most_similar_class(image_embedding, class_files)
    print(f"The most similar class is {most_similar_class} with a score of {score}")
else:
    print("Could not process the image as no cars were detected.")
