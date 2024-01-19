from python import Python


fn main() raises:
    let os = Python.import_module("os")
    let bn = Python.import_module("builtins")
    let glob = Python.import_module("glob")
    let pathLib = Python.import_module("pathlib")
    let pydub = Python.import_module("pydub")
    let np = Python.import_module("numpy")
    let math = Python.import_module("math")
    let librosa = Python.import_module("librosa")

    let CHUNK_LENGTH_MS = 3000
    let SPECTROGRAM_PATH = os.path.join(".", "dataset", "spectrograms3sec")
    let GENRE_ANNOTATIONS_PATH = os.path.join(
        ".", "giantsteps-key-dataset", "annotations", "genre"
    )
    let AUDIO_FILES_PATH = os.path.join(".", "giantsteps-key-dataset", "audio")

    fn get_genre(audio_file_id: PythonObject) raises -> PythonObject:
        let genre_annotation_path = os.path.join(
            GENRE_ANNOTATIONS_PATH, audio_file_id + ".genre"
        )
        let file = bn.open(genre_annotation_path)
        let genre = file.read().strip()
        _ = file.close()
        return genre

    var audio_files = glob.glob(AUDIO_FILES_PATH + "/*.wav")

    for audio_file in audio_files:
        let audio_file_id = pathLib.Path(audio_file).stem
        print(audio_file_id)

        let genre_annotation_path = os.path.join(
            GENRE_ANNOTATIONS_PATH, audio_file_id + ".genre"
        )
        let genre = get_genre(audio_file_id)
        let audio_segment = pydub.AudioSegment.from_wav(audio_file)
        let num_chunks = (bn.len(audio_segment).to_float64() / CHUNK_LENGTH_MS).to_int()

        for i in range(num_chunks):
            let start = i * CHUNK_LENGTH_MS
            let end = (i + 1) * CHUNK_LENGTH_MS
            let chunk = audio_segment.get_sample_slice(start, end)

            # # compute melspectrogram
            var audio_array = np.array(chunk.get_array_of_samples())
            let max_int = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype("float32") / max_int

            let sr = chunk.frame_rate

            # let mel_spec = librosa.feature.melspectrogram(
            #     y=audio_array,
            #     sr=sr,
            #     n_fft=2048,
            #     hop_length=512,
            #     n_mels=128,
            #     fmin=20,
            #     fmax=22050,
            # )
            # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # # save to file
            # np_file_path = os.path.join(SPECTROGRAM_PATH, genre)
            # if not os.path.exists(np_file_path):
            #     os.makedirs(np_file_path)
            # chunk_name = audio_file_id + "." + str(i)
            # np.save(os.path.join(np_file_path, chunk_name), mel_spec_db)
