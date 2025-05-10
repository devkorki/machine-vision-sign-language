#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Setting seed for reproducibility
from numpy.random import seed
seed(10)
tf.random.set_seed(20)

# Load datasets
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

# Separate features and labels
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

Y_test = test["label"]
X_test = test.drop(labels=["label"], axis=1)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape into 28x28x1
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
print(X_train.shape)
print(X_test.shape)

# Display 25 training images
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(Y_train[i])
plt.show()

# Display 25 test images
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10), subplot_kw={'xticks': [], 'yticks': []})
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(Y_test[i])
plt.show()

# Split training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=7)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15
)
datagen.fit(X_train)

# Display shapes
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)

# Build CNN model
model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding='same', input_shape=[28, 28, 1]),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(units=64, activation="relu"),
    layers.Dense(units=26, activation="softmax"),
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fit model using augmented data
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=128),
    validation_data=(X_val, Y_val),
    steps_per_epoch=len(X_train) // 128,
    epochs=50,
    verbose=2
)

# Predict and evaluate
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

print(classification_report(Y_test, predictions))
