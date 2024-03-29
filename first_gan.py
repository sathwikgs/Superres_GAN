# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from tensorflow.python.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import adam
from tensorflow.python.keras.layers import Reshape, Conv2DTranspose, BatchNormalization, Activation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image


num_epochs = 50
batch_size = 32
input_height = 32
input_width = 32
output_height = 256
output_width = 256

train_dir = '/kaggle/input/flower-enhance/train/'
val_dir = '/kaggle/input/flower-enhance/test/'
train_size = len(
    glob.glob(train_dir + "/*-in.jpg"))

small_input_filenames = glob.glob(train_dir + "/*-in.jpg")
large_input_filenames = glob.glob(train_dir + "/*-out.jpg")
# small_images = np.zeros(
#             (train_size, input_width, input_height, 3))
Small_images = np.empty((train_size, 32*32*3), dtype=np.uint8)
Large_images = np.empty((train_size, 256, 256, 3), dtype=np.uint8)

for i in range(train_size):
    small_img = small_input_filenames[i]
    large_img = large_input_filenames[i]
    small_images = np.array(Image.open(small_img))
    small_images = (small_images.astype(np.float32) - 127.5)/127.5
    small_images = small_images.reshape(1, input_width*input_height*3)
    Small_images[i] = small_images
    large_images = np.array(Image.open(large_img))
    large_images = (large_images.astype(np.float32) - 127.5) / 127.5
    # large_images = large_images.reshape(1, input_width * input_height * 3)
    Large_images[i] = large_images


def adam_optimizer():

    return adam(lr=0.0002, beta_1=0.5)


z_dim = input_height*input_width*3


def create_generator():
    model = Sequential()

    # Reshape input into 32x32x256 tensor via a fully connected layer
    model.add(Dense(64 * 32 * 32, input_dim=z_dim))
    model.add(Reshape((32, 32, 64)))
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


def create_discriminator():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=(256, 256, 3)))
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


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        small_images = np.zeros(
            (batch_size, 32, 32, 3))
        large_images = np.zeros(
            (batch_size, 256, 256, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        small_images = small_images.reshape(batch_size, 32*32*3)    
        yield (small_images, large_images)
        counter += batch_size
        
        
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=32*32*3)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def plot_generated_images(train_dir, epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    img_gen_object = image_generator(100, train_dir)
    small_images, large_images = next(img_gen_object)
    noise = small_images[np.random.randint(low=0, high=small_images.shape[0], size=examples)]
    generated_images = generator.predict(noise)*255
    generated_images = generated_images.reshape(100, 256, 256, 3)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)


def training(train_dir, epochs=1, batch_size=32):
    # Loading the data
    # (X_train, y_train, X_test, y_test) = load_data()
    # batch_count = X_train.shape[0] / batch_size
    

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    for e in range(1, epochs + 1):
        img_gen_object = image_generator(batch_size, train_dir)
        small_images, large_images = next(img_gen_object)
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_size)):
            # generate  random noise as an input  to  initialize the  generator
            noise = small_images[np.random.randint(low=0, high=small_images.shape[0], size=batch_size)]

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            image_batch = large_images[np.random.randint(low=0, high=large_images.shape[0], size=batch_size)]

            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Tricking the noised input of the Generator as real data
            noise = small_images[np.random.randint(low=0, high=small_images.shape[0], size=batch_size)]
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(train_dir, e, generator)


training(train_dir, 50, 32)
