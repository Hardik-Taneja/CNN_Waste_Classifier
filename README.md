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

# Week-2 Progress:
Key Updates & Improvements:

**1. Built & Improved CNN Model:** <br>
I added Batch Normalization after each convolutional block to improve stability and speed up training. Additionally, I used He Uniform Initialization to set better initial weights, preventing issues like vanishing gradients.

**2. Optimized Feature Extraction & Regularization:** <br>
The model's convolutional layers were enhanced by progressively increasing the number of filters (32 → 64 → 128 → 256) to capture both low-level and high-level image features. I also applied Dropout (0.5) in the fully connected layers to prevent overfitting and ensure the model generalizes well on new data.

**3. Fixed Data Pipeline & Image Preprocessing:** <br>
I implemented ImageDataGenerator to apply real-time image augmentation, including rotation, zoom, width & height shift, shear, and horizontal flipping. These transformations make the model more robust to variations in input images.

**4. Added Early Stopping & Model Checkpoint:** <br>
To prevent overfitting, I implemented EarlyStopping, which stops training if the validation loss does not improve for a set number of epochs.
Additionally, I used ModelCheckpoint to automatically save the best-performing model during training, ensuring that I retain the most optimal version for further evaluation.
