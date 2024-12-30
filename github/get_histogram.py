import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def generate_histograms(input_directory="Dynamic_ses", output_directory="Histograms"):
    """
    Belirtilen dizindeki ses dosyalarından histogramları oluşturur ve çıktı dizinine kaydeder.

    :param input_directory: Ses dosyalarının bulunduğu dizin
    :param output_directory: Histogramların kaydedileceği dizin
    """
    # Çıktıların kaydedileceği dizini oluştur
    os.makedirs(output_directory, exist_ok=True)

    # Ses dosyalarını işleme
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".wav"):  # Sadece .wav uzantılı dosyaları işle
            file_path = os.path.join(input_directory, file_name)

            try:
                # Ses dosyasını yükle
                y, sr = librosa.load(file_path)

                # Zaman alanında dalga formu histogramını oluştur
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                librosa.display.waveshow(y, sr=sr)
                plt.title(f"Waveform of {file_name}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")

                # Mel-spectrogram oluştur ve görselleştir
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                plt.subplot(2, 1, 2)
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"Mel-Spectrogram of {file_name}")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")

                # Görüntüyü kaydet
                output_file = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_histogram.png")
                plt.savefig(output_file)
                plt.close()
                print(f"{file_name} için histogram başarıyla oluşturuldu ve {output_file} dosyasına kaydedildi.")

            except Exception as e:
                print(f"Hata oluştu: {file_name} - {str(e)}")



