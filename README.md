<div align="center">
<h1>Brain Tumor Classification And Segmentation Using GradCAM</h1>
<img src="./assets/project-image.webp" 
width=50%
alt="Brain Tumor Classification And Segmentation"
style="padding:5px">
</div>
This project aims to employ the Gradient-Weighted Class
Activation Mapping(GradCAM) algorithm to understand how
Convolution Neural Networks(CNN's) make the decisions
they do by visualizing the feature extraction of the 
neural network.

## Language Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Libraries Used

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

# About The Dataset:

[Brain Tumor MRI Images](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

This is a dataset containing a collection of 7023 MRI scans of the human brain in JPG format. The scans belong to 4 different diagnosis classes:
Gliomas, Meningiomas, Pituitary and no Tumors. The class distribution is fairly balanced, which eliminates the need of
oversampling/undersampling to counter class imbalances. However, some images had to be removed due to duplication.

### 1) Training Set Class Distribution

<img src ="assets/trainDistribution.png" height="70%" style="padding-left: 15px">

### 2) Testing Set Class Distribution

<img src ="assets/testDistribution.png" height="70%" style="padding-left: 15px">

### 3) Validation Set Class Distribution

<img src ="assets/valDistribution.png" height="70%" style="padding-left: 15px">

# Installation:

To setup this project, you'll need
