Smart Violence Detection
AI – Enhanced Surveillance System
1. Overview

Smart Violence Detection is an AI-based surveillance system that identifies violent activities in real time from video streams or CCTV footage. It uses deep learning and computer vision to detect aggressive actions and improve safety monitoring in public and private areas.

The system is built with OpenCV, TensorFlow/Keras, and Streamlit, forming a complete workflow from data preprocessing and model training to live detection and visualization.

2. Objectives

Build a deep learning model that detects violence in real-time video feeds.

Design an easy-to-use interface for testing and demonstrations.

Improve the response efficiency of surveillance systems using AI automation.

Combine AI, computer vision, and web technologies in a single project.

3. Technologies Used
Category	Tools and Frameworks
Programming Language	Python
Machine Learning	TensorFlow, Keras, NumPy, Scikit-learn
Computer Vision	OpenCV
Frontend Interface	Streamlit
Database	MongoDB (optional)
Version Control	Git, GitHub
4. Features

Detects violent actions in videos or webcam feeds.

Deep learning model built using convolutional neural networks.

Streamlit interface for live and recorded video testing.

Works with both pre-recorded and real-time camera inputs.

Secure optional integration with MongoDB for data handling.

Modular structure with separate files for training, evaluation, and inference.

5. Project Structure
Smart-Violence-Detection/
│
├── app.py                     # Core application logic
├── app_streamlit.py           # Streamlit interface
├── dataset_loader.py          # Dataset handling
├── db.py                      # Database connection (optional)
├── evaluate.py                # Model evaluation
├── inference.py               # Model inference
├── model.py                   # Model architecture
├── preprocess.py              # Frame preprocessing
├── realtime_detection.py      # Real-time detection
├── train.py                   # Model training
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignore rules
└── README.md                  # Documentation

6. Installation and Setup

Step 1: Clone the repository

git clone https://github.com/shonu01/Smart-Violence-Detection.git
cd Smart-Violence-Detection


Step 2: Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac


Step 3: Install dependencies

pip install -r requirements.txt


Step 4: Create a .env file in the project root (if using database or API keys)

MONGO_URI=your_mongodb_connection_string
OPENAI_API_KEY=your_api_key_if_applicable


Step 5: Run the application

streamlit run app_streamlit.py

7. Usage

To train the model:

python train.py


To evaluate the model:

python evaluate.py


To run real-time detection:

python realtime_detection.py

8. Results and Performance
Metric	Value (Example)
Accuracy	93%
Precision	0.91
Recall	0.89
F1-Score	0.90
Inference Speed	20–25 FPS

(Replace with your actual results if available.)

9. Future Enhancements

Integration with cloud-based vision APIs for improved scalability.

Real-time alerts through email or SMS when violence is detected.

Edge deployment on devices like Raspberry Pi or Jetson Nano.

Web dashboard for admin monitoring and analytics.

10. Author

Name: Shonu
Role: MCA Graduate
GitHub: https://github.com/shonu01
