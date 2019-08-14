from tensorflow.python.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import adam
from tensorflow.python.keras.layers import Reshape, Conv2DTranspose, BatchNormalization, Activation

import numpy as np
import glob
from PIL import Image


num_epochs = 50
batch_size = 32
input_height = 32
input_width = 32
output_height = 256
output_width = 256

train_dir = 'train'
train_size = len(
    glob.glob(train_dir + "/*-in.jpg"))

input_filenames = glob.glob(train_dir + "/*-in.jpg")
# small_images = np.zeros(
#             (train_size, input_width, input_height, 3))
images = []
for i in range(train_size):
    img = input_filenames[i]
    small_images = np.array(Image.open(img))
    small_images = (small_images.astype(np.float32) - 127.5)/127.5
    small_images = small_images.reshape(1,input_width*input_height*3)
    images.append(small_images)


def adam_optimizer():

    return adam(lr=0.0002, beta_1=0.5)


z_dim = input_height*input_width*3


def generator():
    model = Sequential()

    # Reshape input into 32x32x256 tensor via a fully connected layer
    model.add(Dense(256 * 32 * 32, input_dim=z_dim))
    model.add(Reshape((32, 32, 256)))
    model.add(Conv2DTranspose(                  # Transposed convolution layer, from 32x32x256 into 64x64x128 tensor
        128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())             # Batch normalization
    model.add(LeakyReLU(alpha=0.01))            # Leaky ReLU
    model.add(Conv2DTranspose(                  # Transposed convolution layer, from 64x64x128 to 128x128x64 tensor
        64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())             # Batch normalization
    model.add(LeakyReLU(alpha=0.01))            # Leaky ReLU
    model.add(Conv2DTranspose(                  # Transposed convolution layer, from 128x128x64 to 256x256x3 tensor
        3, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))               # Tanh activation
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    model.summary()
    return model


def discriminator():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    model.summary()
    return model


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(32, 32, 3))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


