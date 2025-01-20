# Breast Ultrasound-Based Deep Learning Project

## **Project Overview**
This repository contains the implementation of a deep learning-based framework for breast cancer diagnosis using ultrasound imaging. The primary aim of this project is to develop robust models for:
- **Semantic Segmentation:** Identifying and segmenting tumor regions in breast ultrasound (BUS) images.
- **Classification:** Classifying the identified tumors as either benign or malignant.
- **Multi-task Learning:** Combining segmentation and classification tasks into a single unified model.

### **Problem Statement**
Breast cancer remains a leading cause of mortality worldwide, and early detection is critical for improving survival rates. Ultrasound imaging, though widely used, is prone to interpretation errors due to its complexity. This project leverages deep learning to enhance the diagnostic process by automating tumor detection and classification.

---

## **Notebook Structure**
The project notebook is divided into the following sections:

### **1. Introduction**
- Problem definition and objectives.
- Discussion on the importance of automated diagnostic tools.
- Overview of challenges in using BUS images for diagnosis.

### **2. Data Loading**
- Installation of necessary libraries (e.g., PyTorch, albumentations).
- Loading the dataset from the provided repository.
- Description of dataset structure, including benign and malignant tumor classes, and masks for segmentation.

### **3. Exploratory Data Analysis (EDA)**
- Visualization of BUS images and their corresponding masks.
- Analysis of class distribution (benign vs. malignant).
- Pixel-level distribution (tumor vs. background).
- Key observations and challenges identified during EDA.

### **4. Data Preprocessing**
- **Normalization:** Scaling image pixel values for efficient training.
- **Resizing:** Standardizing image dimensions for neural network input.
- **Data Balancing:** Addressing class imbalance through data augmentation techniques such as flips, rotations, and translations.
- **Mask Handling:** Resolving issues with multiple masks per image by either merging masks or duplicating images.

### **5. Model Development**
#### **Segmentation Model**
- Semantic segmentation using pre-trained models from `segmentation_models_pytorch`.
- Metrics: Dice coefficient and Intersection over Union (IoU).

#### **Classification Model**
- Binary classification using convolutional neural networks (CNNs).
- Metrics: Accuracy, Precision, Recall, and F1-Score.

#### **Multi-task Model**
- Unified architecture combining segmentation and classification tasks.
- Metrics: Mean Average Precision (mAP) and Mean IoU.

### **6. Training and Evaluation**
- **Training Process:** Splitting the dataset into training, validation, and test sets. Hyperparameter tuning and optimization techniques.
- **Evaluation Metrics:** Detailed analysis using metrics and confusion matrices for each model.

### **7. Results and Comparisons**
- Performance analysis of segmentation, classification, and multi-task models.
- Comparative summary highlighting the strengths and limitations of each approach.

---

## **Authors**
- Ranquel Monllor Parres [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/raquel-monllor-parres/)
- Victor Company Rubio [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/victor-company-rubio-bb287b25b/)
- Jimmy Al Khawand 
- Santiago Moreno Pineda [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/santiago-moreno-pineda-6a814a231/)

---

## **How to Use This Repository**
1. Upload the jupyter notebook `Breast_ultrasound_project.ipynb` to your Drive account and open it in Google Colab or clone this repository to your local machine:
   ```bash
   https://github.com/smoreno18/Breast_Ultrasound_project.git
   ```

2. Upload to your Drive the model's weights in order to avoid the training step (Optional).

  
4. Run the notebook step by step to reproduce the results or modify it for further experimentation.
   `NOTE:` If you clone the repository you must do some modifications in order to adjust the script to your local enviroment.
---



## **Acknowledgments**
- Libraries: PyTorch, albumentations, segmentation_models_pytorch.
- Special thanks to the contributors and researchers who made this project possible.

