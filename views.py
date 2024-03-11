from flask import Flask, render_template, redirect, url_for, request
import librosa
import joblib
import numpy as np
import soundfile



labels = ['Old_Fear', 'Old_Pleasant-surprise', 'Old_Sad', 'Old_angry',
       'Old_disgust', 'Old_happy', 'Old_neutral', 'Young_angry',
       'Young_disgust', 'Young_fear', 'Young_happy', 'Young_neutral',
       'Young_pleasant_surprised', 'Young_sad']

def extract_features(audio_path):
    results = np.array([])
    with soundfile.SoundFile(audio_path) as file:
        X = file.read(dtype='float32')
        sr = file.samplerate
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr).T, axis=0)
    results = np.hstack((results, mfccs))
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sr))
    results = np.hstack((results, chroma))
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr))
    results = np.hstack((results, mel))
    return results
        

model = joblib.load('./Model/Emotion_Detector_Audio.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio = request.files['audio']
        file_name = audio.filename
        audio.save(f'./static/uploads/{file_name}')

        test = []

        test.append(extract_features(f'./static/uploads/{file_name}'))

        test = np.array(test)

        pred = model.predict(test)

        label = labels[np.argmax(pred)]

        test = []

        return render_template('index.html', label=label)

    return render_template('index.html')