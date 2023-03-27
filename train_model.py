import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_model():
    # Set the path of the npy files
    dataset_path = "./dataset/spectrograms3sec"

    # Get the labels from the folder names
    genres = os.listdir(dataset_path)
    num_genres = len(genres)
    print(f"num_genres: {num_genres}")

    # Extract features for all audio files and create X and y arrays
    print("Loading data")
    X = []
    y = []
    for i, genre in enumerate(genres):
        genre_path = os.path.join(dataset_path, genre)
        for npy_file in os.listdir(genre_path):
            npy_file_path = os.path.join(genre_path, npy_file)
            spectrogram = np.load(npy_file_path)
            X.append(spectrogram)
            y.append(i)

    print("Converting to numpy arrays")
    X = np.array(X)
    y = np.array(y)

    print("Split data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Build model")
    input_shape = (X_train.shape[1], X_train.shape[2], 1) # add channel dimension for Conv2D
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_genres, activation='softmax'))

    print("Compile model")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train model")
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1), 
        y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1), y_test))


if __name__ == "__main__":
    train_model()
