# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load data
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# Get training labels
labels = train['label'].values

# Plot the quantities in each class
plt.figure(figsize=(18, 8))
sns.countplot(x=labels)
plt.show()

# Drop labels from training data
train.drop('label', axis=1, inplace=True)

# Prepare image data
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

# One-hot encode labels
label_Binarizer = LabelBinarizer()
labels = label_Binarizer.fit_transform(labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

# Scale and reshape image data
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(24, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)

# Save model
model.save("sign_mnist_cnn_50_Epochs.h5")
print("Model saved")

# Plot training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()

# Prepare test data
test_labels = test['label']
test.drop('label', axis=1, inplace=True)

test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_images = test_images / 255
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Predict on test data
y_pred = model.predict(test_images)

# Convert true and predicted labels to class indices
if len(test_labels.shape) > 1:
    true_labels = np.argmax(test_labels, axis=1)  
else:
    true_labels = test_labels

if len(y_pred.shape) > 1:
    predicted_labels = np.argmax(y_pred, axis=1)  
else:
    predicted_labels = y_pred

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# DICTIONARY
number_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Real-time ASL recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Define region of interest
    roi = frame[100:400, 320:620]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # Display ROI
    cv2.imshow('ROI (scaled and gray)', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 2)

    # Prepare ROI for prediction
    roi = roi.reshape(1, 28, 28, 1) / 255.0
    prediction = np.argmax(model.predict(roi), axis=-1)[0]
    predicted_letter = number_to_letter.get(prediction, "Unknown")

    # Display prediction
    cv2.putText(copy, f"Prediction: {predicted_letter}", (320, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live ASL Recognition', copy)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()




# Create directory to store images
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else:
        pass
