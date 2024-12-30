import os
import librosa
import soundfile as sf
from pydub import AudioSegment
from k_means_96acc_ver2 import predict_speak  # predict_speak fonksiyonunun olduğu dosya

def run(input_directory="full_sound_file", output_directory="böl_ses", combined_directory="tanınmış_sesler"):
    #.wav dosyasını yükleme
    input_file = os.path.join(input_directory, "full_sound.wav")

    # Çıktı dizinlerini oluşturma (eğer mevcut değilse)
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(combined_directory, exist_ok=True)

    # Ses dosyasını yükleme
    audio, sr = librosa.load(input_file, sr=None)

    # Ses dosyasını 2 saniyelik parçalara bölme
    chunk_duration = 2  # saniye olarak
    chunk_samples = chunk_duration * sr
    total_samples = len(audio)

    for i in range(0, total_samples, chunk_samples):
        chunk = audio[i:i + chunk_samples]
        output_file = os.path.join(output_directory, f"chunk_{i // chunk_samples + 1}.wav")
        sf.write(output_file, chunk, sr)

    print("Ses dosyası parçalama tamamlandı ve dosyalar kaydedildi.")

    # Tahminlere göre ses dosyalarını birleştirme
    speaker_chunks = {}

    # Parçalanan dosyalar için tahmin işlemi
    for chunk_file in sorted(os.listdir(output_directory)):
        if chunk_file.endswith(".wav"):
            chunk_path = os.path.join(output_directory, chunk_file)
            predicted_speaker = predict_speak(chunk_path)
            print(f"Dosya: {chunk_file}, Tahmin Edilen Konuşmacı: {predicted_speaker}")

            if predicted_speaker not in speaker_chunks:
                speaker_chunks[predicted_speaker] = []
            speaker_chunks[predicted_speaker].append(chunk_path)

    # Her konuşmacının segmentlerini birleştir ve kaydet
    for speaker, chunks in speaker_chunks.items():
        combined_audio = AudioSegment.empty()

        for chunk_path in chunks:
            chunk_audio = AudioSegment.from_wav(chunk_path)
            combined_audio += chunk_audio

        combined_file = os.path.join(combined_directory, f"{speaker}.wav")
        combined_audio.export(combined_file, format="wav")
        print(f"Konuşmacı: {speaker}, Birleştirilmiş dosya: {combined_file}")

        # Kullanılan chunk dosyalarını sil
        for chunk_path in chunks:
            os.remove(chunk_path)
            print(f"Silindi: {chunk_path}")

    # Orijinal ses dosyasını sil
    os.remove(input_file)
    print(f"Orijinal dosya silindi: {input_file}")

    # 3 saniyeden kısa ses dosyalarını sil
    for combined_file in os.listdir(combined_directory):
        file_path = os.path.join(combined_directory, combined_file)
        if combined_file.endswith(".wav"):
            audio = AudioSegment.from_wav(file_path)
            duration_seconds = len(audio) / 1000  # milisaniyeyi saniyeye çevir
            if duration_seconds < 3:
                os.remove(file_path)
                print(f"Silindi (3 saniyeden kısa): {file_path}")

    print("Tüm işlemler tamamlandı ve geçici dosyalar silindi.")
