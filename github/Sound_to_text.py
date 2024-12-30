import os
from pydub import AudioSegment
import speech_recognition as sr
import requests

def extract_text_from_wav(wav_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            print(f"Ses dosyası işleniyor: {wav_path}...")
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="tr-TR")
        return text
    except sr.UnknownValueError:
        return "Ses dosyasından metin çıkartılamadı: Anlaşılamayan içerik."
    except sr.RequestError as e:
        return f"API veya hizmet hatası: {e}"
    except Exception as e:
        return f"Bir hata oluştu: {e}"

def convert_to_pcm_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav", codec="pcm_s16le")
        return output_path
    except Exception as e:
        print(f"Dönüştürme hatası: {e}")
        return None

def translate_text(text, api_key):
    try:
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {
            "q": text,
            "source": "tr",
            "target": "en",
            "key": api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            return result["data"]["translations"][0]["translatedText"]
        else:
            return f"Çeviri API hatası: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Çeviri sırasında hata: {e}"

def determine_topic_from_text_using_google_api_key(text, api_key):
    try:
        if len(text.split()) < 20:
            return "Metin çok kısa"

        url = "https://language.googleapis.com/v1/documents:classifyText"
        headers = {"Content-Type": "application/json"}
        payload = {
            "document": {
                "type": "PLAIN_TEXT",
                "content": text
            }
        }
        params = {"key": api_key}
        response = requests.post(url, headers=headers, json=payload, params=params)
        if response.status_code == 200:
            result = response.json()
            if "categories" in result and result["categories"]:
                return result["categories"][0]["name"]
            else:
                return "Konu belirlenemedi"
        else:
            return f"Google API hatası: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Google API ile konu belirleme sırasında hata: {e}"

def process_all_wav_files_in_directory(directory, api_key):
    results = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            input_file = os.path.join(directory, file_name)
            converted_file = os.path.join(directory, "converted_" + file_name)

            # WAV dosyasını PCM formatına dönüştür
            convert_to_pcm_wav(input_file, converted_file)

            # Metin çıkarma
            text = extract_text_from_wav(converted_file)
            print(text)
            if text and not text.startswith("Ses dosyasından metin çıkartılamadı"):
                # Metni İngilizceye çevir
                translated_text = translate_text(text, api_key)

                # Kelime sayısını kontrol et
                word_count = len(text.split())
                if word_count < 20:
                    print(f"Metin çok kısa, atlanıyor: {translated_text}")
                    results[file_name] = {"topic": "Metin çok kısa", "word_count": word_count}
                else:
                    # Konu belirleme
                    topic = determine_topic_from_text_using_google_api_key(translated_text, api_key)

                    # Sonuçları sözlüğe ekle
                    results[file_name] = {"topic": topic, "word_count": word_count}

            # Geçici dosyayı sil
            if os.path.exists(converted_file):
                os.remove(converted_file)

    # Tüm sonuçları yazdır
    return results


def sound_to_text_func():
    # Kullanım
    api_key = "AIzaSyAqfvQBH3FAlLKic5ZUyfZaoxbBe31cIJ0"  # Google Cloud API Anahtarı
    path = "tanınmış_sesler"
    result = process_all_wav_files_in_directory(path, api_key)

    # İşlem sonrası klasördeki dosyaları sil
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Silindi: {file_path}")
        except Exception as e:
            print(f"Silme sırasında hata: {e}")

    print(result)
    return result