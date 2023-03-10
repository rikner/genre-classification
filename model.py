import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
from keras.preprocessing.image import ImageDataGenerator


def GenreModel(input_shape=(288, 432, 4), classes=9):

  X_input = Input(input_shape)

  X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1)))(X)
  X=BatchNormalization(axis = 3)(X)
  X=Activation('relu')(X)
  X=MaxPooling2D((2, 2))(X)

  X=Conv2D(64, kernel_size = (3, 3), strides = (1, 1))(X)
  X=BatchNormalization(axis = -1)(X)
  X=Activation('relu')(X)
  X=MaxPooling2D((2, 2))(X)

  X=Conv2D(128, kernel_size = (3, 3), strides = (1, 1))(X)
  X=BatchNormalization(axis = -1)(X)
  X=Activation('relu')(X)
  X=MaxPooling2D((2, 2))(X)


  X=Flatten()(X)

  X=Dropout(rate = 0.3)

  X=Dense(classes, activation = 'softmax', name = 'fc' + str(classes))(X)

  model=Model(inputs = X_input, outputs = X, name = 'GenreModel')

  return model
