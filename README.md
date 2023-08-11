# Genre Classification Neural Network Prototype
This repository trains a neural network for music genre classification.
It's currently based on the GiantSteps key dataset (https://github.com/GiantSteps/giantsteps-key-dataset).

## How To

Install dependencies:
> brew install ffmpeg
> brew install sox
> python3 -m pip install -r requirements.txt

Get the giantsteps key dataset:
> git submodule update --init --recursive 

> cd giantsteps-key-dataset

> ./audio_dl.sh

> ./convert_audio.sh

> cd ..


Create dataset for training:
> python3 feature_extraction.py

Train network:
> python3 train_model.py 

If you want to speed up training on Apple Silicon, refer to:
> https://developer.apple.com/metal/tensorflow-plugin/
