# Sign Language Recognition with CNN and SVM

This project implements a system to recognize American Sign Language (ASL) letters using computer vision techniques and machine learning models. It combines deep learning (CNN) on the Sign Language MNIST dataset with traditional SVM classification using segmented images from a custom grid.

## Project Structure

```
├── CNN_Model.py                 # CNN training and evaluation using Sign Language MNIST
├── video_test.py               # Real-time webcam prediction using trained CNN
├── Preprocessing_LastFile.py   # Grid image preprocessing and SVM classification
├── sign_mnist_train.csv        # Dataset file (download from Kaggle)
├── sign_mnist_test.csv         # Dataset file (download from Kaggle)
├── amer_sign2.png              # Custom image with hand signs in grid layout
├── enhanced_processed_letters/ # Directory with enhanced letter images
└── README.md
```

## Models

### CNN (Convolutional Neural Network)

- Implemented using TensorFlow/Keras
- Trained on the Sign Language MNIST dataset
- Includes dropout, batch normalization, and multiple convolutional layers
- Optionally enhanced with real-time data augmentation via `ImageDataGenerator`

### SVM (Support Vector Machine)

- Implemented using scikit-learn
- Trained on enhanced images cropped from `amer_sign2.png`
- Uses preprocessing steps like CLAHE, adaptive thresholding, and morphological operations
- Useful for small-scale recognition without large datasets

## Real-Time Recognition

The `video_test.py` script enables real-time hand sign recognition via webcam:

- Captures a region of interest (ROI) from the live camera feed
- Converts it to grayscale and resizes it to 28x28 pixels
- Passes it to the trained CNN model
- Displays predicted letter on screen

To exit the camera feed, press `ESC`.

## Dataset

This project uses the official [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle.

### Manual Download

1. Go to the Kaggle dataset page:  
   https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. Download the dataset and extract:
   - `sign_mnist_train.csv`
   - `sign_mnist_test.csv`
3. Place both files in the root of this repository.

### Download with Kaggle API

If you have your Kaggle API credentials set up:

```bash
pip install kaggle

kaggle datasets download -d datamunge/sign-language-mnist
unzip sign-language-mnist.zip
```

## Requirements

Install required Python libraries:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
tensorflow
opencv-python
matplotlib
numpy
pandas
seaborn
scikit-learn
```

## Data Augmentation (Optional)

To improve generalization during CNN training, enable real-time data augmentation with:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15
)
datagen.fit(X_train)

model.fit(datagen.flow(X_train, Y_train, batch_size=128), ...)
```

This augments the training data on-the-fly with small transformations.

## License

This project is intended for academic use. Please cite the original Kaggle dataset if used for publication or external research.
