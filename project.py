# Speech Emotion Recognition Model (Using RAVDESS Dataset)

import pandas as pd
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# FOLDER PATH OF DATASET
dataset_path = "AI ML LEARN\emotion.py\RAVDESS"

# MAPING EMOTIONS
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# FUNCTION TO EXTRACT FEATURES FROM AN AUDIO FILE 
def extract_feature(file_path):
    audio , sr = librosa.load(file_path , sr=None , res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y = audio , sr = sr , n_mfcc=40) # WE FIND MFCC TO DETECT SOUND QUALITY , 40 PROVIDE A GOOD BALANCE AND GOOD AT MANAGING DATA
    mfcc_mean = np.mean(mfcc.T , axis = 0)
    return mfcc_mean

# FUNCTION TO LOAD AUDIO FILES FROM ROOT 
def load_files(dataset_path):
    feature = []
    label = []

    for root , dir , files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root , file)

                emotion_code = file.split('-')[2]
                emotion = emotion_map.get(emotion_code , 'unknown')

                features = extract_feature(file_path)

                feature.append(features)
                label.append(emotion)

    return np.array(feature) , np.array(label)
X, y = load_files(dataset_path)

# ENCODING LABELS USING LABEL ENCODER
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# SCALING USING STANDARD SCALER
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# SPLITTING DATA FOR TRAINING AND TESTING
X_train , X_test , y_train , y_test = train_test_split(x_scaled , y_encoded , test_size=0.2 , random_state=42)

# FITTING DATA ON MODEL
model = SVC(kernel='rbf' , C=10 , gamma=0.01 , random_state=42)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test , y_pred , target_names=encoder.classes_))

# SAVING MODEL
joblib.dump(model , 'emotion_model.pkl')
joblib.dump(encoder , 'encoder.pkl')
joblib.dump(scaler , 'scaler.pkl')

def final_predict(audio_path):
    # LOADING MODEL
    model = joblib.load('emotion_model.pkl')
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    features = extract_feature(audio_path)
    feature_2d = features.reshape(1,-1)
    scaler_feature = scaler.transform(feature_2d)


    prediction = model.predict(scaler_feature)
    return encoder.inverse_transform(prediction)[0]

audio_path = r"AI ML LEARN\emotion.py\03-01-07-01-02-02-02.wav"
# audio_path = input("Enter The Path Of Audio : ")

if not os.path.exists(audio_path):
    print(f"Error: File not found at {audio_path}")
else:
    print(final_predict(audio_path))