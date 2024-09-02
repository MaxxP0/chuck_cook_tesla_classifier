import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = []
        self.labels = []
        for folder in os.listdir(root):
            for image in os.listdir(os.path.join(root, folder)):
                self.images.append(os.path.join(root, folder, image))
                self.labels.append(folder)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = F.to_tensor(image)  # Convert to tensor here
        label = self.labels[idx]
        return image, label

def extract_cars(model, images, labels, threshold=0.7, device='cuda'):
    model.eval()
    cars = []
    car_labels = []
    try:
        with torch.no_grad():
            predictions = model(images.to(device))
        
        for i, prediction in enumerate(predictions):
            boxes = prediction['boxes']
            scores = prediction['scores']
            pred_labels = prediction['labels']
            
            car_indices = (pred_labels == 3) & (scores > threshold)  # 3 is the COCO class index for 'car'
            car_boxes = boxes[car_indices]
            
            if len(car_boxes) > 0:
                image = images[i]
                for box in car_boxes:
                    x1, y1, x2, y2 = box.int()
                    car = image[:, y1:y2, x1:x2]
                    cars.append(car)
                    car_labels.append(labels[i])
    except RuntimeError as e:
        print(f"Error processing batch: {e}")
        return [], []
    
    return cars, car_labels

def main():
    # Set up CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up the dataset and dataloader
    dataset = ImageDataset(r'D:\AllDatasets\ChuckTesla\Resized')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)  # Reduced batch size

    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model = model.to(device)
    model.eval()

    # Process batches and extract cars
    all_cars = []
    all_labels = []
    for batch_images, batch_labels in tqdm(dataloader, desc="Processing images"):
        print(f"Processing batch of size {batch_images.shape[0]}")
        cars, labels = extract_cars(model, batch_images, batch_labels, device=device)
        all_cars.extend(cars)
        all_labels.extend(labels)
        print(f"Extracted {len(cars)} cars from this batch")

    # Save extracted car images in class-specific folders
    base_dir = 'extracted_cars'
    os.makedirs(base_dir, exist_ok=True)
    
    print("Saving extracted car images...")
    for i, (car, label) in enumerate(tqdm(zip(all_cars, all_labels), total=len(all_cars))):
        class_dir = os.path.join(base_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        plt.imsave(f'{class_dir}/car_{i}.png', car.permute(1, 2, 0).cpu().numpy())

    print(f"Extracted {len(all_cars)} car images across {len(set(all_labels))} classes.")

if __name__ == "__main__":
    main()
