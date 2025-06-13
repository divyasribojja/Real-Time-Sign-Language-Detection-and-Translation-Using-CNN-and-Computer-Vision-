📘 Real-Time Sign Language Detection and Translation using CNN & Computer Vision

📌 Overview
This project uses Convolutional Neural Networks (CNN) and Computer Vision (OpenCV) to detect hand gestures representing American Sign Language (ASL) letters in real-time through a webcam. It translates these gestures into corresponding alphabet characters, assisting in communication with the hearing or speech impaired.

🧠 Features
Real-time webcam-based hand gesture recognition

CNN model trained on ASL alphabet dataset

Translates signs into English alphabets (A-Z)

Displays prediction confidence

Highlighted Region of Interest (ROI) for better UX

🛠️ Tech Stack
Technology	Role
Python	Core programming language
TensorFlow / Keras	Deep Learning (CNN model)
OpenCV	Real-time video processing
NumPy	Numerical operations
ASL Dataset	For training gesture classification

🧾 How to Run
Clone the Repository


git clone ```https://github.com/divyasribojja/Real-Time-Sign-Language-Detection-and-Translation-Using-CNN-and-Computer-Vision-.git ```
cd sign-language-detection
Install Dependencies

pip install -r requirements.txt
Run the Application


python sign_language_detection.py
Make sure your webcam is connected. Press q to exit the window.

🏋️‍♀️ Dataset
We used the ASL Alphabet Dataset to train our CNN model on 26 English letters (A-Z).
                  # (optional) Training dataset directory
📈 Results
Achieved ~95% accuracy on test set

Real-time prediction speed: ~25 FPS

High confidence detection in consistent lighting

📸 Sample Output
Frame	Detected Sign
Sign: A (97.2%)

🚀 Future Improvements
Integrate word/phrase level translation

Add sentence formation and voice output

Improve gesture segmentation using background subtraction
