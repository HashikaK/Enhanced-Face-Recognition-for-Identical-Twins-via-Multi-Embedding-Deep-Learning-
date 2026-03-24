# Enhanced Face Recognition for Identical Twins via Multi-Embedding Deep Learning

## Overview

This project presents a deep learning-based face recognition system specifically designed to distinguish **identical twins**, a challenging problem due to their high facial similarity.

The system leverages:

* **MTCNN** for face detection
* **InceptionResnetV1 (FaceNet)** for feature extraction
* **Multi-Embedding Validation** to improve recognition robustness

---

## Key Features

* Multi-embedding storage per individual
* Cosine similarity-based matching
* Margin-based ambiguity detection for twins
* Real-time webcam recognition
* Image upload-based verification
* SQLite database for embedding storage

---

## Methodology

1. Face Detection using MTCNN
2. Feature Extraction using FaceNet
3. Multiple embeddings stored per person
4. Matching using cosine similarity
5. Decision based on:

   * Similarity threshold
   * Margin between top matches

---

## Dataset Structure

```
dataset/train/
    twin1_A/
        img1.jpg
        img2.jpg
    twin1_B/
        img1.jpg
        img2.jpg
```

Each person should have **multiple images (recommended: 5+)**.

---

## Installation

```bash
git clone https://github.com/your-username/twin-face-recognition.git
cd twin-face-recognition
pip install -r requirements.txt
```

---

##  How to Run

### Run Main Application

```bash
python app/main.py
```

### Access:

* Home: http://localhost:5000/
* Upload: http://localhost:5000/upload
* Webcam: http://localhost:5000/webcam

---

## Output

* Recognized person name with similarity score
* Ambiguous detection for twins when margin is low
* Unknown classification for low similarity

---

##  Reproducibility

* Pretrained model: VGGFace2 (FaceNet)
* Automatic dataset loading into SQLite database
* Deterministic embedding comparison via cosine similarity

---

## Code Availability

This code is directly associated with the manuscript submitted to *The Visual Computer*.
Please cite the paper if you use this work.

---

## Future Work

* ROC curve evaluation
* Larger twin datasets
* Transformer-based embeddings
* Deployment optimization (TensorRT)

---

##  Author

Hashika K, Harismita V, Swetha S 

---
