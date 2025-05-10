# Sign Language Recognition with CNN and SVM

This project implements a system to recognize American Sign Language (ASL) letters using computer vision techniques and machine learning models. It combines deep learning (CNN) on the Sign Language MNIST dataset with traditional SVM classification using segmented images from a custom grid, and adds a second pipeline that trains a CNN on actual image data enhanced through computer vision preprocessing and augmentation.

---

## Project Structure
```
├── CNN_Model.py                 # CNN training using Sign Language MNIST (CSV)
├── Preprocessing_LastFile.py   # Processes amer_sign2.png into enhanced letter images
├── image_cnn_from_png.py       # Augments real images and trains a CNN from folders
├── webcam_test_custom.py       # Live webcam ASL prediction using custom-trained model
├── video_test.py               # Webcam prediction using MNIST-trained CNN
├── amer_sign2.png              # Grid of ASL hand signs used in preprocessing
├── enhanced_processed_letters/ # Output: processed letter images (A-Y)
├── custom_dataset/             # Output: folder-structured dataset ready for CNN
├── README.md
```

---

## Models

### CNN (Convolutional Neural Network)
- Implemented using TensorFlow/Keras
- Two versions:
  - Trained on `sign_mnist_train.csv`
  - Trained on real processed image folders (`custom_dataset/`)
- Includes dropout, multiple `Conv2D` and pooling layers
- Enhanced with `ImageDataGenerator` for augmentation

### SVM (Support Vector Machine)
- Implemented using scikit-learn
- Trained on preprocessed letter images extracted from `amer_sign2.png`
- Applies CLAHE, adaptive thresholding, and dilation

---

## Real-Time Recognition

Two webcam scripts:

- `video_test.py`: Predicts from MNIST-trained CNN model
- `webcam_test_custom.py`: Predicts from model trained on real images

Each script:
- Captures webcam input
- Extracts a hand-sized region of interest (ROI)
- Resizes and normalizes to 28×28 grayscale
- Predicts and overlays the ASL letter

Press **ESC** to exit.

---

## Dataset

This project uses the official [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle.

### Manual Download
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
2. Download and extract:
   - `sign_mnist_train.csv`
   - `sign_mnist_test.csv`
3. Place both files in the root of this repository.

### Kaggle API (Optional)
```bash
pip install kaggle
kaggle datasets download -d datamunge/sign-language-mnist
unzip sign-language-mnist.zip
```

---

## Requirements
Install all dependencies:
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

---



## License
This project is intended for academic and educational use. Please cite the original [Kaggle dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) if used in research.
