# Siamese Network for Face Recognition

## Overview

This repository contains a Jupyter notebook (`Siamese_Network.ipynb`) implementing a Siamese Network for face recognition. The project demonstrates how to train a neural network to determine whether two facial images belong to the same person using a contrastive loss function. The implementation uses the Olivetti Faces dataset, with data augmentation to create training pairs.

## Features

- **Custom Data Loader**: The `Data_loader` class handles dataset extraction, creation, and augmentation, including generating positive/negative image pairs.
- **Siamese Network Architecture**: Built with TensorFlow/Keras, using twin subnetworks that process images and compute similarity via L1 distance.
- **Contrastive Loss**: Trains the network to minimize distance between similar images and maximize distance between dissimilar ones.
- **Visualization Tools**: Includes functions to display sample image pairs for dataset inspection.
- **Training Pipeline**: Complete workflow from data preparation to model training and evaluation.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Albumentations

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn albumentations
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/siamese-network-face-recognition.git
cd siamese-network-face-recognition
```

2. Run the Jupyter notebook:

```bash
jupyter notebook Siamese_Network_4.ipynb
```

3. Follow the notebook to:
   - Load and preprocess the dataset
   - Generate image pairs
   - Build and train the Siamese Network
   - Evaluate model performance

## Implementation Details

The project demonstrates:

- Data augmentation techniques using Albumentations
- Custom layer implementations (L1 distance)
- Training with contrastive loss
- Visualization of results
