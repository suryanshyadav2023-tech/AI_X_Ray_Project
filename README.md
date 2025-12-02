🩺 Chest X-Ray Disease Classifier — AI Project
Deep Learning Model for Multi-Label Medical Image Classification

Author: Suryansh Yadav (23BCE0581)
VIT Vellore

Model files - https://drive.google.com/file/d/133XZHLd-QsUFQcMiUB8YzgfYtaxMsC29/view?usp=drive_link

📌 Project Overview

This project builds an AI model capable of detecting multiple chest diseases from a single X-ray image.
Using the NIH ChestX-ray14 dataset, the system learns to classify 14 thoracic conditions using deep learning.

The core model used is DenseNet121, well-known for its strong feature extraction capability in medical imaging tasks.

🎯 Problem Statement

To design and develop a deep-learning model that analyzes chest X-ray images and predicts the probability of multiple co-existing diseases.
Since an X-ray can contain more than one abnormality, this is treated as a multi-label classification problem.

🧠 AI Model Architecture
DenseNet121

121-layer convolutional neural network

Pretrained on ImageNet, then fine-tuned for medical imaging

Dense connections ensure efficient feature reuse

Performs extremely well on tasks requiring recognition of subtle patterns

Output Layer

14 output nodes

Sigmoid activation (gives probability 0–1 for each disease)

Each disease is treated independently

Diseases Predicted:
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,Pleural_thickening, Pneumonia, Pneumothorax.
🧪 Dataset
NIH ChestX-ray14

112,120 X-rays

Multi-label format

Contains metadata + disease annotations

Highly imbalanced dataset

Preprocessing

Images resized to 224×224 pixels (DenseNet requirement)

Pixel normalization

Data augmentation (flip, rotate)

Grayscale converted to RGB (3 channels)

⚙️ Training Details

Loss function: Binary Cross Entropy (BCE)

Activation: Sigmoid

Optimizer: Adam

Evaluation metrics:

AUC (primary metric for medical tasks)

Accuracy (less relevant for multi-label)

Regularization:

Dropout

Data Augmentation

Early Stopping

🔍 How the Model Classifies Diseases

Takes the full X-ray image (not divided manually)

Applies convolution filters that “scan” the image

Extracts features like edges, shadows, blobs

Dense connections share learned features across layers

Fully connected layer outputs 14 probability scores

Values > 0.5 indicate disease presence

🌐 API Overview (AI Perspective)

The AI model is accessed through a simple Flask REST API.

Endpoint: /predict

Accepts an image

Preprocesses it

Runs DenseNet121 on the input

Returns disease probabilities as JSON

🖥️ Web Interface

A lightweight front-end allows:

Uploading X-ray images

Displaying disease probabilities

Highlighting top predictions

The front-end communicates with the AI model through an API call.

🚀 Future Enhancements (AI Perspective)

Add Grad-CAM heatmaps to show why the model predicted a disease

Use stronger models like DenseNet201, EfficientNet, or Vision Transformers

Improve training with:

Class balancing

Advanced augmentations

Contrastive learning

Expand to detect disease severity.
Train on more diverse datasets to reduce bias

📌 Conclusion

This project demonstrates the capability of convolutional neural networks to interpret medical images and support diagnostic decisions.
DenseNet121 performs efficiently in multi-label classification, and the accompanying web interface makes the system usable for real-world testing and demonstration.
