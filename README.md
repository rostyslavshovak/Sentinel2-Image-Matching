# **Computer vision. Sentinel-2 image matching Using U-Net and ORB Algorithm**

## **Overview**

This project focuses on matching satellite images. The goal is to identify images of the same regions or highly similar regions captured at different times or seasons.

This project implements a hybrid approach to match Sentinel-2 satellite images using a combination of a **U-Net model** for feature extraction and the **ORB (Oriented FAST and Rotated BRIEF)** algorithm for image matching. Sentinel-2 satellite images are preprocessed, passed through the **U-Net model** to extract features, and compared using **cosine similarity** to find matching images. The ORB algorithm is used to visually highlight similar regions in the matched image pairs.

## **Data Collection and preproccesing**

Data for this project was sourced from the Kaggle [Deforestation in Ukraine from Sentinel2 dataset](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine).
The main idea of this dataset is to identify the regions of deforestation in Kharkiv, Ukraine from the Sentinel2 satellite imagery.

The dataset consists of images stored in .jp2 format.

- **Preprocessing**:
  - The `.jp2` images are resized to `256x256` to fit the input size of the U-Net model.
  - The images are converted to grayscale, normalized, and stored in the `processed_dataset` for further processing.

- **Preprocessing Steps**:
1. Convert `.jp2` files to `.jpg` format.
2. Resize and normalize the images.
3. Save the preprocessed images for further analysis.

## **Model Architecture**

The project utilizes a **U-Net model** for feature extraction from the images. The U-Net architecture consists of several convolutional layers and pooling operations to capture image features.

- **Input**: Grayscale satellite images (converted from the original RGB).
- **Output**: Feature maps representing the high-level features of the image.

The **ORB algorithm** is used to detect and match keypoints between pairs of images, helping in visualizing similarities between them.

## **Image Matching Process**

**U-Net Feature Extraction**:
1. Load Image: The images are preprocessed to match the input size of the U-Net model (`256x256`).
2. Feature Extraction: The U-Net model extracts high-level features from each image.
3. Cosine Similarity: The features of different images are compared using **cosine similarity** to identify similar image pairs.

**ORB Feature Matching**:
Once similar image pairs are identified:
1. ORB Keypoint Detection: The ORB algorithm detects keypoints in both images.
2. Feature Matching: Keypoints are matched using a **Brute Force Matcher**.
3. Visualization: Matched areas between the two images are displayed.

## **Model Inference**

To perform inference and match images:

1. **Preprocess your dataset** using the following commands from:
   ```python
   dataset_creatinon.py
   ```
   This will preprocess the raw Sentinel-2 images and save them in the specified folder.
   
2. **Run Feature Extraction and Similarity Matching**: Use the following function to find similar image pairs:
   ```python
   find_similar_images(your_path, model, threshold=0.75)
   ```
   This will compare all images from your directory and output pairs of matched images with a similarity above the specified threshold (suggest to use above 0.75).
  
3. **Visualize Matched Areas**: To visualize the ORB-matched regions between two images, use:
   ```python
   draw_orb_matches(image_path1, image_path2, max_lines=15)
   ```
## Usage
Steps to run inference:

1. **Clone the Repository**:
  ```bash
  https://github.com/rostyslavshovak/Sentinel2-Image-Matching.git
  cd sentinel2-image-matching
  ```

2. **Install Dependencies**: Ensure that all necessary dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

3. **Continue with Model inference steps and check for comments inside** `sentinel2-unet_demo.ipynb` 

