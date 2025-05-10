# image_cnn_from_png.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Constants
INPUT_DIR = "enhanced_processed_letters"
AUGMENTED_DIR = "custom_dataset"
CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 letters, excluding J and Z

# Step 1: Create folders for custom dataset
for letter in CLASSES:
    os.makedirs(os.path.join(AUGMENTED_DIR, letter), exist_ok=True)

# Step 2: Define image augmentation functions
def augment_image(image):
    augmented = []
    image = cv2.resize(image, (28, 28))
    
    # Rotation
    for angle in [-15, -10, 0, 10, 15]:
        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (28, 28))
        augmented.append(rotated)

    # Flipping
    augmented.append(cv2.flip(image, 1))  # horizontal

    # Brightness
    brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
    augmented.extend([brighter, darker])

    return augmented

# Step 3: Generate and save augmented images
for letter in CLASSES:
    img_path = os.path.join(INPUT_DIR, f"{letter}_enhanced.png")
    if not os.path.exists(img_path):
        continue

    base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    augmented_images = augment_image(base_img)

    for i, img in enumerate(augmented_images):
        save_path = os.path.join(AUGMENTED_DIR, letter, f"{letter}_{i}.png")
        cv2.imwrite(save_path, img)

print("Augmented dataset saved in custom_dataset/")

# Step 4: Load and train model using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    AUGMENTED_DIR,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    AUGMENTED_DIR,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Step 5: Define CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(24, activation='softmax'))  # 24 classes

# Step 6: Compile and train
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# Step 7: Save the model
model.save("cnn_custom_from_png.h5")
print("Model trained and saved as cnn_custom_from_png.h5")
