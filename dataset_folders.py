from glob import glob
from os.path import basename, join
import os

dataset_path = "./dataset/"
spectrogram_path = join(dataset_path, "spectrograms3sec")

def create_dataset_folders():
    os.makedirs(spectrogram_path)

    audio_files = glob("./giantsteps-key-dataset/audio/*.wav")

    audio_file_ids = [basename(file).split(".")[0] for file in audio_files]

    id_genres = []
    for id in audio_file_ids:
        genre_annotation_path = join(
            ".", "giantsteps-key-dataset", "annotations", "genre", id + ".LOFI.genre")

        with open(genre_annotation_path) as f:
            genre = f.read().strip()
            id_genres.append((id, genre))

    unique_genres = set([id_genre[1] for id_genre in id_genres])

    genre_counts = []
    for genre in unique_genres:
        ids_for_genre = [id_g[0] for id_g in id_genres if id_g[1] == genre]
        genre_counts.append((genre, len(ids_for_genre)))
        print(f'{genre}: {len(ids_for_genre)}')
        os.makedirs(os.path.join(spectrogram_path, f'{genre}'))


if __name__ == "__main__":
    create_dataset_folders()
