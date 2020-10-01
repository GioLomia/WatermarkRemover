import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt
import datetime
# from autoencoder import Denoise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

class ConvNetAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(ConvNetAutoEncoder, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.maxp1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.maxp2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        
        self.encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        
        self.conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsample1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv5 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsample2 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.upsample3 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        
    
    def call(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.encoded(x)
        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.upsample2(x)
        x = self.conv6(x)
        x = self.upsample3(x)
        x = self.conv7(x)
        return x

def loss(x, x_bar):
    return tf.losses.mean_squared_error(x, x_bar)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction\


if __name__ == '__main__':

  (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = tf.reshape(x_train, (len(x_train), 28, 28, 1))
  x_test = tf.reshape(x_test, (len(x_test), 28, 28, 1))

  noise_factor = 0.5
  # x_train_noisy = x_train + noise_factor * tf.random_normal(shape=x_train.shape) 
  # x_test_noisy = x_test + noise_factor * tf.random_normal(shape=x_test.shape)

  x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
  x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

  print(x_train)
  model = ConvNetAutoEncoder()

  # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  # global_step = tf.Variable(0)

  num_epochs = 50
  batch_size = 4

  # for epoch in range(num_epochs):
  #     print("Epoch: ", epoch)
  #     for x in range(0, len(x_train), batch_size):
  #         x_inp_noisy = x_train_noisy[x : x + batch_size]
  #         x_inp = x_train[x : x + batch_size]
  #         loss_value, grads, reconstruction = grad(model, x_inp_noisy, x_inp)
  #         optimizer.apply_gradients(zip(grads, model.trainable_variables),
  #                               global_step)
          
  #         if global_step.numpy() % 200 == 0:
  #             print("Step: {},         Loss: {}".format(global_step.numpy(),
  #                                           loss(x_inp, reconstruction).numpy()))