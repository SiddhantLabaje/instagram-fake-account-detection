# Instagram Fake Account Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-green?logo=render)

A machine learning web application that detects whether an Instagram account is **fake or real** using a trained neural network model. Built with Flask, TensorFlow, and deployed on Render.

🔗 **Live Demo:** [https://instagram-fake-account-detection-utus.onrender.com](https://instagram-fake-account-detection-utus.onrender.com)

---

## Features

- Detects fake Instagram accounts using a deep learning model
- Fetches real-time profile data via **Instaloader**
- Falls back to dataset-based features if profile is private or unavailable
- Displays prediction result with confidence percentage
- Clean, responsive UI built with Tailwind CSS
- Dockerized and deployed on Render

---

## Tech Stack

| Layer        | Technology                          |
|--------------|--------------------------------------|
| Backend      | Python, Flask                        |
| ML Model     | TensorFlow / Keras (Neural Network)  |
| Data Prep    | Pandas, NumPy, Scikit-learn, SMOTE   |
| Profile Fetch| Instaloader                          |
| Frontend     | HTML, Tailwind CSS                   |
| Deployment   | Docker, Render                       |

---

## Project Structure

```
Fake_account_detection_using_ml/
│
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── combine_csv.py          # Merges train.csv and test.csv
├── instagram.py            # Instagram utility helpers
│
├── templates/
│   ├── index.html          # Home page
│   ├── result.html         # Prediction result page
│   └── error.html          # Error page
│
├── models/
│   ├── instagram_model.h5  # Trained Keras model
│   └── scaler.pkl          # Fitted StandardScaler
│
├── plots/                  # Training evaluation plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── instagram.csv           # Combined dataset
│
├── Dockerfile              # Docker configuration
├── .dockerignore
└── requirements.txt
```

---

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/) and contains the following features:

| Feature                | Description                                      |
|------------------------|--------------------------------------------------|
| `profile pic`          | Whether the account has a profile picture (0/1)  |
| `nums/length username` | Ratio of digits to total username length         |
| `fullname words`       | Number of words in the full name                 |
| `nums/length fullname` | Ratio of digits to full name length              |
| `name==username`       | Whether full name matches username (0/1)         |
| `description length`   | Length of the bio/description                    |
| `external URL`         | Whether an external URL is present (0/1)         |
| `private`              | Whether the account is private (0/1)             |
| `#posts`               | Total number of posts                            |
| `#followers`           | Number of followers                              |
| `#follows`             | Number of accounts followed                      |
| `fake`                 | Target label — 1 = Fake, 0 = Real                |

---

## Model Architecture

- Input layer → 11 features
- Dense(64) + BatchNormalization + Dropout(0.3)
- Dense(128) + BatchNormalization + Dropout(0.4)
- Dense(64) + BatchNormalization + Dropout(0.3)
- Output: Dense(2, softmax)

Trained with:
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Class imbalance handled with **SMOTE**
- Early stopping + model checkpointing

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Fake_account_detection_using_ml.git
cd Fake_account_detection_using_ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (skip if models/ already exists)

```bash
python train_model.py
```

### 4. Start the Flask app

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## Run with Docker

```bash
docker build -t fake-account-detection .
docker run -p 5000:5000 fake-account-detection
```

Then open `http://localhost:5000`

---

## How It Works

1. User enters an Instagram username on the home page
2. App tries to fetch the real profile using **Instaloader**
3. If the profile is accessible, real features are extracted
4. If not (private/unavailable), a fallback from the dataset is used
5. Features are scaled and passed to the trained neural network
6. Result is displayed with a **Fake / Real** label and confidence score

---

## Author

**Sanket Sawant**

---

## License

This project is for educational purposes only.
