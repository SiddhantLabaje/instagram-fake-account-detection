from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import re
import random
import instaloader
import os

app = Flask(__name__)

# Load model and scaler
try:
    model = load_model('models/instagram_model.h5')
    scaler = joblib.load('models/scaler.pkl')

    print("Model and scaler loaded successfully")

except Exception as e:

    print(f"Error loading model or scaler: {e}")

    model = None
    scaler = None

# Load dataset
try:
    dataset = pd.read_csv('instagram.csv')

    feature_columns = [
        'profile pic',
        'nums/length username',
        'fullname words',
        'nums/length fullname',
        'name==username',
        'description length',
        'external URL',
        'private',
        '#posts',
        '#followers',
        '#follows'
    ]

    dataset_features = dataset[feature_columns]

    print("Dataset loaded successfully")

except Exception as e:

    print(f"Error loading dataset: {e}")

    dataset_features = None

# Initialize Instaloader
L = instaloader.Instaloader()

# Validate username
def is_valid_instagram_username(username):

    pattern = r'^[a-zA-Z0-9._]{1,30}$'

    return bool(re.match(pattern, username))

# Random fallback features
def get_random_features():

    random_row = dataset_features.sample(
        n=1,
        random_state=random.randint(0, 10000)
    )

    return random_row

# Fetch Instagram profile
def get_profile_features(username):

    try:

        profile = instaloader.Profile.from_username(
            L.context,
            username
        )

        features = {

            'profile pic':
                1 if profile.profile_pic_url else 0,

            'nums/length username':
                sum(c.isdigit() for c in profile.username)
                / len(profile.username),

            'fullname words':
                len(profile.full_name.split()),

            'nums/length fullname':
                sum(c.isdigit() for c in profile.full_name)
                / (
                    len(profile.full_name)
                    if profile.full_name else 1
                ),

            'name==username':
                int(
                    profile.full_name.lower()
                    == profile.username.lower()
                ),

            'description length':
                len(profile.biography),

            'external URL':
                1 if profile.external_url else 0,

            'private':
                int(profile.is_private),

            '#posts':
                profile.mediacount,

            '#followers':
                profile.followers,

            '#follows':
                profile.followees
        }

        features_df = pd.DataFrame([features])

        return features_df

    except Exception as e:

        print(f"Error fetching profile for {username}: {e}")

        return None

# Home route
@app.route('/')
def index():

    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    if model is None or scaler is None:

        return render_template(
            'error.html',
            error="Model or scaler failed to load."
        )

    try:

        username = request.form['username'].strip()

        if not is_valid_instagram_username(username):

            raise ValueError(
                "Invalid Instagram username."
            )

        features_df = get_profile_features(username)

        # Fallback if profile fetch fails
        if features_df is None:

            if dataset_features is None:

                raise ValueError(
                    "Dataset unavailable."
                )

            features_df = get_random_features()

            note = (
                "Real profile could not be fetched. "
                "Using fallback dataset features."
            )

        else:

            note = (
                "Prediction based on real-time "
                "Instagram profile data."
            )

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict
        prediction = model.predict(features_scaled)

        confidence = float(prediction[0][1])

        is_fake = confidence > 0.5

        confidence_percentage = round(confidence * 100, 2)

        return render_template(
            'result.html',
            username=username,
            is_fake=is_fake,
            confidence=confidence_percentage,
            note=note
        )

    except Exception as e:

        return render_template(
            'error.html',
            error=str(e)
        )

# Run Flask app
if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )