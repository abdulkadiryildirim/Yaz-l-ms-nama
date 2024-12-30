import os
import librosa
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from pydub import AudioSegment
from scipy.spatial.distance import cdist
import joblib  # Model kaydetme/yükleme için
import warnings
import json
import mimetypes

warnings.filterwarnings("ignore")

# Logger ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Eğitim ve tahmin işlemleri için gerekli dosya yolları
MODEL_FILE = "kmeans_model.pkl"
SCALER_FILE = "scaler.pkl"
SPEAKERS_FILE = "speakers.json"
MAPPING_FILE = "mapping.json"

# Statik ses dosyası yolu
DATA_DIR = "Dynamic_ses"

# Model ve scaler kaydetme/yükleme
def save_model_and_speakers(kmeans_model, scaler, speakers, mapping):
    """Model, scaler, konuşmacı listesi ve eşleştirme tablosunu kaydet."""
    try:
        joblib.dump(kmeans_model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        with open(SPEAKERS_FILE, 'w') as f:
            json.dump(speakers, f)
        with open(MAPPING_FILE, 'w') as f:
            json.dump({int(k): v for k, v in mapping.items()}, f)  # Anahtarları int formatına dönüştür
        logger.info("Model, scaler, konuşmacı listesi ve eşleştirme tablosu başarıyla kaydedildi.")
    except Exception as e:
        logger.error(f"Model, konuşmacı listesi veya eşleştirme tablosu kaydedilirken hata: {e}")

def load_model_and_speakers():
    """Model, scaler, konuşmacı listesi ve eşleştirme tablosunu yükle."""
    try:
        kmeans_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(SPEAKERS_FILE, 'r') as f:
            speakers = json.load(f)
        with open(MAPPING_FILE, 'r') as f:
            mapping = {int(k): v for k, v in json.load(f).items()}  # Anahtarları int formatına dönüştür
        logger.info("Model, scaler, konuşmacı listesi ve eşleştirme tablosu başarıyla yüklendi.")
        return kmeans_model, scaler, speakers, mapping
    except Exception as e:
        logger.error(f"Model, konuşmacı listesi veya eşleştirme tablosu yüklenirken hata: {e}")
        return None, None, [], {}

# Segmentleme ve özellik çıkarımı

def bol_audio(file_path, segment_duration):
    """Ses dosyasını segmentlere böler."""
    y, sr = librosa.load(file_path)
    segment_samples = segment_duration * sr
    segments = []

    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]
        if len(segment) == segment_samples:  # Eksik segmentleri hariç tut
            segments.append(segment)

    return segments, sr


def ozellik_cikar(audio, sr):
    """Verilen ses segmentinden özellik çıkarır."""
    try:
        # MFCC özellikleri
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=64)
        mfcc_means = mfcc.mean(axis=1)
        mfcc_vars = mfcc.var(axis=1)  # MFCC varyansı

        # Tempo
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

        # Sıfır geçiş oranı (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=audio).mean()

        # Spektral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()

        # Spektral bant genişliği
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()

        # Chroma özellikleri
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).mean()

        # Enerji
        energy = np.sum(audio ** 2) / len(audio)

        # Enerji Entropisi
        frame_length = 2048  # Örnek çerçeve uzunluğu
        hop_length = 512  # Örnek adım uzunluğu
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy_frames = np.sum(frames ** 2, axis=0)
        energy_entropy = -np.sum(energy_frames * np.log2(energy_frames + 1e-10))

        # Tüm özellikleri birleştir
        return np.concatenate([
            mfcc_means,
            mfcc_vars,
            [
                tempo,
                zcr,
                spectral_centroid,
                spectral_bandwidth,
                chroma,
                energy,
                energy_entropy
            ]
        ])
    except Exception as e:
        logger.error(f"Özellik çıkarma sırasında hata: {e}")
        raise e


# Model eğitimi

def train():
    """Modeli eğitir ve doğruluk hesaplayarak gerekli objeleri kaydeder."""
    print("MODEL EGITIMINE BASLANDI")
    try:
        audio_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith(".wav")]

        X = []
        y = []

        for file in audio_files:
            label = os.path.basename(file).split("_")[0]  # Örnek: "label_ses1.wav"
            segments, sr = bol_audio(file, segment_duration=2)

            for segment in segments:
                features = ozellik_cikar(segment, sr)  # Yeni özellik çıkarma fonksiyonunu kullanıyoruz
                X.append(features)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Özellik ölçekleme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float64)

        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        # K-Means eğitimi
        kmeans = KMeans(n_clusters=len(set(y)), n_init=20, max_iter=500, random_state=42)
        kmeans.fit(X_train)

        # Test seti üzerindeki sonuçları değerlendirme
        kmeans_labels = kmeans.predict(X_test)
        cm = confusion_matrix(y_test, kmeans_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {int(col_ind[i]): int(row_ind[i]) for i in range(len(row_ind))}  # Anahtarlar ve değerler int
        kmeans_labels_mapped = np.array([mapping[label] for label in kmeans_labels])

        accuracy = np.sum(kmeans_labels_mapped == y_test) / len(y_test)
        logger.info(f"K-Means doğruluğu: {accuracy}")
        # Küme merkezleri arasındaki uzaklıkları hesapla
        cluster_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
        np.fill_diagonal(cluster_distances, np.inf)  # Kendilerine olan uzaklıkları sonsuz yap

        # En yakın kümeleri bul ve yazdır
        min_distance_indices = np.unravel_index(np.argmin(cluster_distances), cluster_distances.shape)
        min_distance = cluster_distances.min()
        logger.info(f"Minimum Küme Mesafesi: {min_distance}")

        # Uzaklıkların dağılımı
        max_distance = cluster_distances[cluster_distances != np.inf].max()
        mean_distance = cluster_distances[cluster_distances != np.inf].mean()
        logger.info(f"Maksimum Küme Mesafesi: {max_distance}")
        logger.info(f"Ortalama Küme Mesafesi: {mean_distance}")

        # En yakın kümelerin bilgisi
        closest_clusters = (label_encoder.inverse_transform([min_distance_indices[0]])[0],
                            label_encoder.inverse_transform([min_distance_indices[1]])[0])
        logger.info(f"En Yakın Kümeler: {closest_clusters}")

        # Model, konuşmacı listesi ve eşleştirme tablosu kaydetme
        save_model_and_speakers(kmeans, scaler, label_encoder.classes_.tolist(), mapping)
        return True, f"Model başarıyla eğitildi. Doğruluk: {accuracy}"

    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata: {e}")
        return False, "Model eğitimi sırasında bir hata oluştu."

    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata: {e}")
        return False, "Model eğitimi sırasında bir hata oluştu."

def predict_speak(filepath):
    """Verilen ses dosyasını tahmin eder."""
    try:
        kmeans, scaler, speakers, mapping = load_model_and_speakers()
        if not kmeans or not speakers:
            return "Model yüklenemedi, lütfen yeniden eğitin."
        y_audio, sr = librosa.load(filepath)
        features = np.array(ozellik_cikar(y_audio, sr), dtype=np.float64)
        features = scaler.transform([features])
        predicted_label = kmeans.predict(features)[0]
        mapped_label = mapping.get(predicted_label, -1)

        if mapped_label == -1:
            return "Tahmin eşleşmesi bulunamadı."

        predicted_speaker = speakers[mapped_label]

        # Sınıf yakınlıklarını hesapla ve yazdır
        distances = kmeans.transform(features)[0]
        probabilities = 1 / (1 + distances)  # Daha düşük mesafe daha yüksek olasılık anlamına gelir
        probabilities /= probabilities.sum()  # Normalize et

        for idx, prob in enumerate(probabilities):
            logger.info(f"Sınıf: {speakers[mapping.get(idx, idx)]}, Olasılık: {prob:.4f}")
        print(predicted_speaker)
        return predicted_speaker
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {e}")
        return "Tahmin sırasında bir hata oluştu."