# **Tesla Embedding Classifier for Chuck Cook**

This project is designed to classify Tesla vehicles using an embedding-based approach. The following scripts are used to process images, generate embeddings, and compare them to classify the car type.

## **Project Overview**

### **1. `extract.py`**
The `extract.py` script extracts crops of cars from the dataset shared in the Discord channel. Note that this process requires some manual labeling due to the presence of multiple cars (both Tesla and non-Tesla) in some frames. All extracted images initially end up in the Tesla folder, so please delete any non-Tesla images to ensure accurate classification.

### **2. `embed.py`**
The `embed.py` script takes the cropped images and generates embeddings for them. These embeddings are saved in separate files based on the car type, allowing for easy retrieval and comparison.

### **3. `compare.py`**
The `compare.py` script handles the classification of new, unseen images. It performs the following steps:

1. Extracts a crop of the car from the image.
2. Generates an embedding for the cropped image.
3. Compares this embedding against all previously saved embeddings.

If the cosine similarity between the embeddings exceeds 0.6, the corresponding class receives a +1 score. The final score for each class is obtained by dividing the total score by the number of comparisons made, yielding a value between 0 and 1. The class with the highest score is considered the correct classification.

## TODO
- [x] Use a bounding box model to automatically extract crops of cars from images.
