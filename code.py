import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------- CONFIG -------- #
MODEL_PATH = 'sign_language_model.h5'  # Use your trained CNN model
CLASSES = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']
IMG_SIZE = 64

# -------- Load Model -------- #
model = load_model(MODEL_PATH)

# -------- Prediction Function -------- #
def predict_sign(img, model):
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    predictions = model.predict(reshaped)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
    return CLASSES[class_id], confidence

# -------- Real-Time Webcam Detection -------- #
cap = cv2.VideoCapture(0)
print("📷 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) box for hand gesture
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # Predict sign
    sign, conf = predict_sign(roi, model)
    cv2.putText(frame, f'Sign: {sign} ({conf*100:.1f}%)', (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display output
    cv2.imshow("Real-Time Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
