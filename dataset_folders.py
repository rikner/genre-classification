from glob import glob
from os.path import basename, join
import os

dataset_path = "./dataset/"
spectrogram_path = join(dataset_path, "spectrograms3sec")
# spectrogram_path_train = join(spectrogramt_path, "train")
# spectrogram_path_test = join(spectrogramt_path, "test")
audio_path = join(dataset_path, "audio3sec")

def main():
    os.makedirs(spectrogram_path)
    # os.makedirs(spectrogram_path_train)
    # os.makedirs(spectrogram_path_test)

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

        path_audio = os.path.join(audio_path, f'{genre}')
        os.makedirs(path_audio)

        os.makedirs(os.path.join(spectrogram_path, f'{genre}'))

        # path_train = os.path.join(spectrogram_path_train, f'{genre}')
        # path_test = os.path.join(spectrogram_path_test, f'{genre}')
        # os.makedirs(path_train)
        # os.makedirs(path_test)


if __name__ == "__main__":
    main()
