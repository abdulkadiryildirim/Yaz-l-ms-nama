import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import json
import warnings

warnings.filterwarnings("ignore")


def extract_features(audio, sr, n_mfcc=13):
    """Ses sinyalinden özellik çıkarır."""
    features = {}

    # MFCC özellikleri
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_{i}"] = np.mean(mfcc[i])

    # Mel özellikleri
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    for i in range(128):
        features[f"mel_{i}"] = np.mean(mel[i])

    # Chroma özellikleri
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    for i in range(12):
        features[f"chroma_{i}"] = np.mean(chroma[i])

    # ZCR ve RMS özellikleri
    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    features["rms"] = np.mean(librosa.feature.rms(y=audio))

    return features


def predict_emotions_in_directory(directory, model_dir):
    results = {}
    try:
        # Model ve dönüştürücüleri yükle
        model = load_model(os.path.join(model_dir, "model_emotion.h5"))
        scaler_mean = np.load(os.path.join(model_dir, "scaler.npy"))
        with open(os.path.join(model_dir, "label_encoder.json"), "r") as f:
            label_classes = json.load(f)

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(label_classes)

        # Dizindeki tüm wav dosyalarını bul
        for file_name in os.listdir(directory):
            if file_name.endswith(".wav"):
                file_path = os.path.join(directory, file_name)

                # Dosya adından kişi adını al (dosya adı: "kadir.wav")
                person_name = os.path.splitext(file_name)[0]

                # Ses dosyasını oku
                audio, sr = librosa.load(file_path, sr=None)

                # Özellik çıkar
                features = extract_features(audio, sr)

                # Modelin beklediği özellikler
                feature_columns = [
                                      f"mfcc_{i}" for i in range(13)
                                  ] + [
                                      f"mel_{i}" for i in range(128)
                                  ] + [
                                      f"chroma_{i}" for i in range(12)
                                  ] + ["zcr", "rms"]

                ordered_features = [features[col] for col in feature_columns if col in features]

                if len(ordered_features) != len(feature_columns):
                    raise ValueError(
                        f"Eksik özellik sayısı: {len(ordered_features)} özellik çıkarıldı, ancak {len(feature_columns)} özellik bekleniyor."
                    )

                features_scaled = (np.array(ordered_features) - scaler_mean)

                # Tahmin yap
                predictions = model.predict(features_scaled.reshape(1, -1))
                predicted_label = np.argmax(predictions, axis=1)[0]

                # Tahmin edilen duygu
                predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]
                results[person_name] = predicted_emotion

    except Exception as e:
        print(f"Hata: {e}")

    return results


def emotion():
    full_sound_file_directory = "tanınmış_sesler"  # Ses dosyalarının bulunduğu ana dizin
    model_directory = "model_emotion"  # Model ve ilgili dosyaların bulunduğu dizin
    emotions = predict_emotions_in_directory(full_sound_file_directory, model_directory)
    print(emotions)
    return emotions

