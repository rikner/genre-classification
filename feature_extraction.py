import librosa
import random
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment, audio_segment
import os
import math
from dataset_folders import audio_path, spectrogram_path
from pathlib import Path

# set chunk length in milliseconds
CHUNK_LENGTH_MS = 3000

GENRE_ANNOTATIONS_PATH = os.path.join(".", "giantsteps-key-dataset", "annotations", "genre")

def main():
    audio_files = glob("./giantsteps-key-dataset/audio/*.wav")
    for audio_file in audio_files:
        # get genre for audio file
        audio_file_id = os.path.basename(audio_file).split(".")[0]
        genre_annotation_path = os.path.join(GENRE_ANNOTATIONS_PATH, audio_file_id + ".LOFI.genre")
        with open(genre_annotation_path) as f:
            genre = f.read().strip()
        
        print(f"chunking and extracting features of {audio_file} ({genre})")

        audio_segment = AudioSegment.from_wav(audio_file)
        num_chunks = math.floor(len(audio_segment) / CHUNK_LENGTH_MS)
        for i in range(num_chunks):
            # create chunk
            start = i * CHUNK_LENGTH_MS
            end = (i + 1) * CHUNK_LENGTH_MS
            chunk = audio_segment[start:end]
            chunk_name = Path(audio_file).stem + "." + str(i)

            # export as wav
            # wav_chunk_path = os.path.join(audio_path, genre, chunk_name + ".wav")
            # chunk.export(wav_chunk_path, format="wav")

            # compute melspectrogram
            audio_array = np.array(chunk.get_array_of_samples())
            max_int = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype('float32') / max_int
            sr = chunk.frame_rate
            mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # plot and save the mel spectrogram
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            png_path = os.path.join(spectrogram_path, genre, chunk_name + ".png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()



def save():
    audio_files = glob("./giantsteps-key-dataset/audio/*.wav")

    random_audio_file = random.choice(audio_files)
    audio_data = librosa.load(random_audio_file, mono=True)
    D = librosa.stft(audio_data[0])
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    (mag, phase) = librosa.magphase(D)
    S_db_2 = librosa.amplitude_to_db(mag, ref=np.max)

    plt.figure()
    librosa.display.specshow(S_db_2)
    plt.show()

    np.savez_compressed("spectrum", S_db)


def load():
    S_db = np.load("./spectrum.npy")

    plt.figure()
    librosa.display.specshow(S_db)
    plt.show()


if __name__ == "__main__":
    main()
