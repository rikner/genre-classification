import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import coremltools as ct


DATASET_PATH = "./dataset/spectrograms3sec"

def train_model():
    genres = os.listdir(DATASET_PATH)
    num_genres = len(genres)

    print("Loading data")
    X = []
    y = []
    for i, genre in enumerate(genres):
        genre_path = os.path.join(DATASET_PATH, genre)
        for npy_file in os.listdir(genre_path):
            npy_file_path = os.path.join(genre_path, npy_file)
            spectrogram = np.load(npy_file_path)
            X.append(spectrogram)
            y.append(i)


    print("Converting integer labels to one-hot encoded labels")
    y = to_categorical(y, num_genres)

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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train model")
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    model.fit(
        X_train_reshaped, 
        y_train, 
        epochs=10,
        batch_size=32, 
        validation_data=(X_test_reshaped, y_test))
    
    print("Save model")
    model.save("genre-classifier")

    print("Convert to Core ML model")
    print(model.summary())
    classifier_config = ct.ClassifierConfig(
        class_labels=genres,
        predicted_feature_name="genre",
    )
    coreml_model = ct.convert(model, classifier_config=classifier_config)
    coreml_model.save('GenreClassifier.mlmodel')


if __name__ == "__main__":
    train_model()
