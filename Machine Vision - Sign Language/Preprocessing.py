#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code processes an image with sign language letters in a grid, preprocecess each letter and then trains SVM classifier to recognize each letter.
It then evaluates the classifier with precision, recall, f1-score values.


Preprocessing includes:
    1. Splitting the input image into grid cells corresponding to individual letters.
    2. Resizing, adaptive thresholding, dilating and eroding, edge detection and enhancing each letter for improved clarity.
    3. Saving processed images to specific directories. (processed_letters, enhanced_processed_letters )
    
Additional preprocessing for letters "B", "R", "D", "I", "O", "U", "V", "G" as hand gestures are more complex
The data are also being augmented to generate additional samples.
    
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

# Load the image
image_path = 'C:/Users/Avocando/Documents/Deree/MASTERS/Machine Vision/GIA TA PAIDIA/amer_sign2.png'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image path does not exist: {image_path}")

gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Define grid dimensions (4 rows x 6 columns for 24 letters)
rows, cols = 4, 6
cell_height = gray_image.shape[0] // rows
cell_width = gray_image.shape[1] // cols

# Prepare output directories
processed_dir = "processed_letters"
enhanced_dir = "enhanced_processed_letters"
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(enhanced_dir, exist_ok=True)

output_dir = "labeled_letters"
os.makedirs(output_dir, exist_ok=True)


# Alphabet labels, excluding J and Z
alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY"


# Process each grid cell to extract, resize, and label the letters
index = 0
for r in range(rows):
    for c in range(cols):
        if index >= len(alphabet):  # Stop after 24 letters
            break

        y_start, y_end = r * cell_height, (r + 1) * cell_height
        x_start, x_end = c * cell_width, (c + 1) * cell_width

        # Extract and resize the letter
        letter = gray_image[y_start:y_end, x_start:x_end]
        letter_resized = cv2.resize(letter, (150, 150))

        # Add the label (e.g., A, B, C) as text onto the image
        labeled_image = cv2.putText(
            letter_resized.copy(),
            alphabet[index],
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255),
            2,
            cv2.LINE_AA,
        )

        # Save the labeled image
        letter_filename = os.path.join(output_dir, f"{alphabet[index]}.png")
        cv2.imwrite(letter_filename, labeled_image)
        index += 1

# Process each grid cell to extract, resize, and label the letters
index = 0
for r in range(rows):
    for c in range(cols):
        if index >= len(alphabet):  # Stop after 24 letters
            break

        y_start, y_end = r * cell_height, (r + 1) * cell_height
        x_start, x_end = c * cell_width, (c + 1) * cell_width

        letter = gray_image[y_start:y_end, x_start:x_end]

        try:
            letter_resized = cv2.resize(letter, (150, 150))
        except Exception as e:
            print(f"Error resizing letter at index {index}: {e}")
            continue

        # Save processed images
        letter_filename = os.path.join(processed_dir, f"{alphabet[index]}.png")
        success = cv2.imwrite(letter_filename, letter_resized)
        if not success:
            print(f"Failed to save image for letter: {alphabet[index]} at {letter_filename}")
        else:
            print(f"Saved image for letter: {alphabet[index]} at {letter_filename}")

        index += 1

# Verify processed images
print(f"Processed images saved in {processed_dir}.")
if not os.listdir(processed_dir):
    print("No images were saved in the processed_letters directory. Check your input and processing steps.")

# Directory for labeled letters
base_labeled_dir = "C:/Users/Avocando/Documents/Deree/MASTERS/Machine Vision/GIA TA PAIDIA/labeled_letters"
if not os.path.exists(base_labeled_dir):
    print(f"Base labeled directory does not exist: {base_labeled_dir}")
letters_to_process = ["B", "R", "D", "I", "O", "U", "V", "G"]
image_paths = {letter: os.path.join(base_labeled_dir, f"{letter}.png") for letter in letters_to_process}

# Function to display images
def display_images(images, titles, cmap="gray"):
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Process specific images
output_dir = "processed_letters_output"
os.makedirs(output_dir, exist_ok=True)

for letter, path in image_paths.items():
    if not os.path.exists(path):
        print(f"Error: File not found for letter {letter}: {path}")
        continue

    # Read the image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image for letter {letter}.")
        continue

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 6
    )

    # Dilation and Erosion
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Canny Edge Detection
    edges = cv2.Canny(eroded, 100, 250)

    # Save processed images
    cv2.imwrite(os.path.join(output_dir, f"{letter}_adaptive_thresh.png"), adaptive_thresh)
    cv2.imwrite(os.path.join(output_dir, f"{letter}_dilated.png"), dilated)
    cv2.imwrite(os.path.join(output_dir, f"{letter}_eroded.png"), eroded)
    cv2.imwrite(os.path.join(output_dir, f"{letter}_edges.png"), edges)

    # Display processing steps
    display_images(
        [img, adaptive_thresh, dilated, eroded, edges],
        [
            f"Original {letter}",
            f"{letter} - Adaptive Threshold",
            f"{letter} - Dilated",
            f"{letter} - Eroded",
            f"{letter} - Canny Edges",
        ],
    )

print(f"Processed images saved in {output_dir}.")

# General preprocessing for all letters
processed_images = []
index = 0
for r in range(rows):
    for c in range(cols):
        if index >= len(alphabet):
            break

        y_start, y_end = r * cell_height, (r + 1) * cell_height
        x_start, x_end = c * cell_width, (c + 1) * cell_width

        letter = gray_image[y_start:y_end, x_start:x_end]

        try:
            letter_resized = cv2.resize(letter, (150, 150))
        except Exception as e:
            print(f"Error resizing letter at index {index}: {e}")
            continue

        # Adaptive Threshold and Morphological Cleaning
        adaptive_thresh = cv2.adaptiveThreshold(
            letter_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        # CLAHE Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)

        letter_filename = os.path.join(enhanced_dir, f"{alphabet[index]}_enhanced.png")
        success = cv2.imwrite(letter_filename, enhanced)
        if not success:
            print(f"Failed to save enhanced image for letter: {alphabet[index]} at {letter_filename}")
        else:
            print(f"Enhanced image saved for letter: {alphabet[index]} at {letter_filename}")

        processed_images.append((alphabet[index], enhanced))
        index += 1

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
for ax, (label, img) in zip(axes.flatten(), processed_images):
    ax.imshow(img, cmap="gray")
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()
plt.show()

# Classification using SVM
data_dirs = ["enhanced_processed_letters", "processed_letters"]
data_dirs = [os.path.abspath("enhanced_processed_letters")]

images, labels = [], []

# # Load images and labels
# for data_dir in data_dirs:
#     for letter in alphabet:
#         path = os.path.join(data_dir, f"{letter}_enhanced.png")
#         if not os.path.exists(path):
#             print(f"Warning: File not found: {path}")
#             continue

#         image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if image is not None:
#             images.append(image.flatten())  # Flatten the image
#             labels.append(letter)

# # Convert to NumPy arrays
# X, y = np.array(images), np.array(labels)

# if X.shape[0] == 0:
#     print(f"No samples found in directories: {data_dirs}. Please check the directories.")
# else:
#     # Check class distribution
#     print("Class distribution in dataset:", Counter(y))

#     # Train-test split with stratification
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, stratify=y, random_state=42
#     )

#     print("Class distribution in training set:", Counter(y_train))
#     print("Class distribution in test set:", Counter(y_test))

#     # Train SVM classifier
#     classifier = SVC(kernel="linear", class_weight="balanced")
#     classifier.fit(X_train, y_train)

#     # Predict and evaluate
#     y_pred = classifier.predict(X_test)

#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
#     print("Confusion Matrix:\n", cm)


# Load images and labels
images, labels = [], []
for data_dir in data_dirs:
    for letter in alphabet:
        path = os.path.join(data_dir, f"{letter}_enhanced.png")
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image.flatten())  # Flatten the image
            labels.append(letter)

# Augment underrepresented classes
augmented_images = []
augmented_labels = []

for image, label in zip(images, labels):
    count = labels.count(label)
    if count == 1:  # Duplicate if only 1 sample
        augmented_images.extend([image, image])  # Add the sample twice
        augmented_labels.extend([label, label])
    else:
        augmented_images.append(image)
        augmented_labels.append(label)

# Update the original lists
images, labels = augmented_images, augmented_labels

# Print the updated class distribution
from collections import Counter
print("Updated class distribution:", Counter(labels))

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.5, stratify=labels, random_state=42
    
    
)



# Train an SVM classifier
classifier = SVC(kernel="linear", class_weight="balanced")
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))



