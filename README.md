Potato Disease Detection using Machine Learning
This project is a machine learning-based solution for detecting potato diseases, specifically Early Blight, Late Blight, and Healthy potato plants. The system utilizes a deep learning model trained on a dataset of potato plant images to classify the health of the plants.

📊 Demo
You can check out the live demo of the Potato Disease Detector on Hugging Face Space.
[Potato Disease Detector Demo on Hugging Face](https://huggingface.co/spaces/Pranav09112002/potato-disease-detector)


🛠️ Technologies Used
Python: Programming language

TensorFlow/Keras: Deep learning framework for training the model

Streamlit: Framework for building the interactive web app

OpenCV: For image processing

Hugging Face Spaces: For easy deployment

GitHub: Code repository

📝 Project Overview
This project focuses on detecting three types of potato plant diseases using a convolutional neural network (CNN). The model takes an input image of a potato leaf and classifies it as:

Healthy – No disease detected.

Early Blight – Early stage of a fungal disease.

Late Blight – Advanced stage of the fungal disease.

The system has been deployed on Hugging Face Spaces for easy access and use.

🧑‍💻 How It Works
Model Training: The model was trained using a dataset of potato plant images, which were preprocessed to resize and normalize the images before feeding them into the neural network.

App Interface: The Streamlit app allows users to upload an image of a potato leaf. The app then uses the trained model to predict whether the plant is Healthy, has Early Blight, or has Late Blight.

Deployment: The web app is deployed on Hugging Face Spaces, where users can upload images and get predictions in real-time.

🤖 Model Details
Model Type: Convolutional Neural Network (CNN)

Training Dataset: Potato plant images categorized as Healthy, Early Blight, and Late Blight.

Input: Image of a potato leaf

Output: Classification label: Healthy, Early Blight, or Late Blight

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

📬 Contact
Author: Pranav

Email: pranavy325@gmail.com
