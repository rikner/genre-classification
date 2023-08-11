import math
import os
from glob import glob
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

CHUNK_LENGTH_MS = 3000

GENRE_ANNOTATIONS_PATH = os.path.join(
    ".", "giantsteps-key-dataset", "annotations", "genre"
)

AUDIO_FILES_PATH = os.path.join(".", "giantsteps-key-dataset", "audio")

SPECTROGRAM_PATH = os.path.join(".", "dataset", "spectrograms3sec")


def feature_extraction():
    audio_files = glob(AUDIO_FILES_PATH + "/*.wav")

    for audio_file in tqdm(audio_files):
        audio_file_id = Path(audio_file).stem
        genre = get_genre(audio_file_id)

        for i, chunk in enumerate(get_chunks(audio_file)):
            # compute melspectrogram
            audio_array = np.array(chunk.get_array_of_samples())
            max_int = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype("float32") / max_int

            sr = chunk.frame_rate
            mel_spec = librosa.feature.melspectrogram(
                y=audio_array,
                sr=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                fmin=20,
                fmax=22050,
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # save to file
            np_file_path = os.path.join(SPECTROGRAM_PATH, genre)
            if not os.path.exists(np_file_path):
                os.makedirs(np_file_path)
            chunk_name = audio_file_id + "." + str(i)
            np.save(os.path.join(np_file_path, chunk_name), mel_spec_db)


def get_genre(audio_file_id):
    genre_annotation_path = os.path.join(
        GENRE_ANNOTATIONS_PATH, audio_file_id + ".genre"
    )
    with open(genre_annotation_path) as f:
        return f.read().strip()


def get_chunks(audio_file):
    audio_segment = AudioSegment.from_wav(audio_file)
    num_chunks = math.floor(len(audio_segment) / CHUNK_LENGTH_MS)
    for i in range(num_chunks):
        start = i * CHUNK_LENGTH_MS
        end = (i + 1) * CHUNK_LENGTH_MS
        yield audio_segment[start:end]


if __name__ == "__main__":
    feature_extraction()
