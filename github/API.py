from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from k_means_96acc_ver2 import predict_speak, train
import böl_tahmin
import duygu_predict
import get_histogram
import Sound_to_text
from flask import Flask, request
from flask_socketio import SocketIO
from urllib.parse import urlparse, parse_qs
import base64
from flask import send_from_directory
import json
import os
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Logger ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Klasör yolu
INPUT_SOUND_DIR = "Dynamic_ses"
os.makedirs(INPUT_SOUND_DIR, exist_ok=True)
PREDICT_SOUND_DIR = "full_sound_file"
os.makedirs(PREDICT_SOUND_DIR, exist_ok=True)
HISTOGRAMS_DIR = "Histograms"
os.makedirs(HISTOGRAMS_DIR, exist_ok=True)

def save_audio_sequentially(file, speaker_name, directory):
    """Ses dosyasını numaralandırılmış bir şekilde kaydeder."""
    try:
        filename = f"{speaker_name}.wav"
        filepath = os.path.join(directory, filename)
        file.save(filepath)
        logger.info(f"Ses dosyası kaydedildi: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Ses dosyası kaydedilirken hata: {e}")
        return None

@app.route('/record', methods=['POST'])
def record_audio():
    """Ses kaydını alır ve kaydeder."""
    speaker_name = request.form.get('speaker_name')
    audio_file = request.files.get('audio')

    if not speaker_name or not audio_file:
        return jsonify({"error": "Konuşmacı adı ve ses dosyası gereklidir."}), 400

    filepath = save_audio_sequentially(audio_file, speaker_name, INPUT_SOUND_DIR)
    if filepath:
        return jsonify({"message": f"Ses dosyası başarıyla kaydedildi: {filepath}"}), 200
    else:
        return jsonify({"error": "Ses dosyası kaydedilemedi."}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Model eğitimi endpoint."""
    success, message = train()
    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 500

@socketio.on('predict')
def handle_prediction_event(data):
    """WebSocket ile tahmin."""
    try:
        logger.info("Tahmin isteği alındı.")

        # Base64 formatındaki ses verisini çöz
        audio_data = data.get("audio_data")
        if not audio_data:
            socketio.emit('prediction', {"error": "Ses verisi alınamadı."})
            return

        # Veriyi bir geçici dosyaya kaydet
        audio_bytes = base64.b64decode(audio_data)
        temp_audio_path = "temp_audio.wav"  # Geçici bir ses dosyası
        with open(temp_audio_path, "wb") as temp_audio_file:
            temp_audio_file.write(audio_bytes)

        # Tahmin fonksiyonunu çağır
        speaker_name = predict_speak(temp_audio_path)
        logger.info(f"Tahmin sonucu: {speaker_name}")
        socketio.emit('prediction', {"speaker": speaker_name})

        # Geçici dosyayı sil
        os.remove(temp_audio_path)
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {e}")
        socketio.emit('prediction', {"error": "Tahmin sırasında bir hata oluştu."})


@app.route('/endfilerecord', methods=['POST'])
def predict_full():
    """Tüm dosya kaydetme."""
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "Ses dosyası gereklidir."}), 400

    filepath = save_audio_sequentially(audio_file, "full_sound", PREDICT_SOUND_DIR)
    if filepath:
        return jsonify({"message": f"Ses dosyası başarıyla kaydedildi: {filepath}"}), 200
    else:
        return jsonify({"error": "Ses dosyası kaydedilemedi."}), 500

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Ses dosyasını analiz eder."""
    try:
        böl_tahmin.run()
        emotion = duygu_predict.emotion()
        topic = Sound_to_text.sound_to_text_func()
        return jsonify({"topic": topic, "emotion": emotion}), 200
    except Exception as e:
        logger.error(f"Ses analiz işlemi sırasında hata: {e}")
        return jsonify({"error": "Ses analiz işlemi sırasında bir hata oluştu."}), 500


@app.route('/histograms/<path:filename>', methods=['GET'])
def serve_histogram_file(filename):
    """Histogram dosyalarını servis eder ve sonrasında siler."""
    try:
        # Eğer filename JSON gibi görünüyorsa, ayrıştır ve sadece 'name' değerini al
        if filename.startswith("{") and filename.endswith("}"):
            parsed_filename = json.loads(filename)
            filename = parsed_filename.get("name", "")

        print(f"HISTOGRAMS_DIR: {os.path.abspath(HISTOGRAMS_DIR)}")
        print(f"Filename: {filename}")

        # Dosya yolunu kontrol et
        file_path = os.path.join(HISTOGRAMS_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Dosya bulunamadı: {file_path}")
            return jsonify({"error": "Histogram dosyası bulunamadı."}), 404

        # Dosyayı gönder
        response = send_from_directory(HISTOGRAMS_DIR, filename)

        # Dosyayı sildikten sonra yanıt dön
        os.remove(file_path)
        print(f"Dosya silindi: {file_path}")
        return response

    except Exception as e:
        logger.error(f"Histogram dosyası servis edilirken hata: {e}")
        return jsonify({"error": "Histogram dosyası servis edilirken bir hata oluştu."}), 500

@app.route('/histograms', methods=['GET'])
def get_histograms():
    get_histogram.generate_histograms()
    """Tüm histogram dosyalarının adlarını ve tam URL'lerini döndürür."""
    try:
        histograms = []
        for file_name in os.listdir(HISTOGRAMS_DIR):
            if file_name.endswith(".png"):
                # Tam URL'yi oluştur
                file_url = f"http://{request.host}/histograms/{file_name}"
                histograms.append({"name": file_name, "url": file_url})

        if not histograms:
            return jsonify({"message": "Histogram dosyası bulunamadı."}), 404

        return jsonify({"histograms": histograms}), 200
    except Exception as e:
        logger.error(f"Histogramlar alınırken hata: {e}")
        return jsonify({"error": "Histogramlar alınırken bir hata oluştu."}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

