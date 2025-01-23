# P4 - To develop a CNN model to classify images of plastic waste

# Dataset
https://www.kaggle.com/datasets/techsash/waste-classification-data/data

# Week-1 Progress: Waste Management Using CNN Model
In the first week of this project, I focused on setting up the pipeline for developing a Convolutional Neural Network (CNN) model to classify waste into categories such as Organic and Recyclable. Here's what was accomplished:

**1. Project Setup:** <br>
Installed required Python libraries, including TensorFlow, OpenCV, NumPy, and Matplotlib, to handle image processing, model creation, and data visualization.
Defined the project's purpose: leveraging CNNs to classify waste images.

**2. Dataset Preparation:** <br>
Organized the dataset into TRAIN and TEST directories for model training and evaluation.
Constructed file paths using Python's os module for dynamic dataset handling.

**3. Image Data Preprocessing:** <br>
Loaded and processed images from the dataset using OpenCV.
Converted images to RGB format and appended them to the x_data list.
Extracted corresponding labels from directory names and stored them in the y_data list.

**4. Data Structuring:** <br>
Created a DataFrame using Pandas to structure the image data (x_data) and their corresponding labels (y_data).
Visualized the distribution of waste categories using a pie chart.

**5. Visualization:** <br>
Generated a pie chart to display the class distribution of the dataset, ensuring balanced data representation for training.
