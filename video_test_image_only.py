# webcam_test_custom.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the custom model trained from real images
model = load_model("cnn_custom_from_png.h5")

# Label map for the 24 ASL letters used
number_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Define region of interest
    roi = frame[100:400, 320:620]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Prepare for prediction
    roi_normalized = roi_resized.reshape(1, 28, 28, 1) / 255.0
    prediction = np.argmax(model.predict(roi_normalized), axis=-1)[0]
    predicted_letter = number_to_letter.get(prediction, "Unknown")

    # Display prediction and bounding box
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 2)
    cv2.putText(copy, f"Prediction: {predicted_letter}", (320, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frames
    cv2.imshow('ROI (grayscale)', roi_resized)
    cv2.imshow('Live ASL Recognition - Custom Model', copy)

    if cv2.waitKey(1) == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
