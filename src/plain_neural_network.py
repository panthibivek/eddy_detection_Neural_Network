#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:20:18 2022

@author: bivek
"""
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Activation, SeparableConv2D, Conv2DTranspose


class plain_net_eddy(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img, target_img):
        self.batch_size = batch_size
        self.img_size = img_size
        #self.data_x = xr.open_dataset(input_img)
        #self.data_y = xr.open_dataset(target_img)
        self.data_x = input_img
        self.data_y = target_img

    def __len__(self):
        return len(self.data_y) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img = self.data_x[i : i + self.batch_size]
        batch_target_img = self.data_y[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for i in range(self.batch_size):
            x[i] = np.expand_dims(batch_input_img[i], 2)
            y[i] = np.expand_dims(batch_target_img[i], 2)
        """
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        """
        return x, y

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###
    X = inputs
    # Entry block
    #X = Conv2D(32, 3, strides=2, padding="same")(inputs)
    #X = Activation("relu")(X)
    #X = BatchNormalization()(X)

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [16,16,16]:
        X = SeparableConv2D(filters, 3, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
        
        X = SeparableConv2D(filters, 3, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)

        X = MaxPooling2D(3, strides=2, padding="same")(X)

    ### [Second half of the network: upsampling inputs] ###

    for filters in [16, 16, 16]:
        X = Conv2DTranspose(filters, 3, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)

        X = Conv2DTranspose(filters, 3, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)

        X = UpSampling2D(2)(X)

    # Add a per-pixel classification layer
    
    X = Conv2DTranspose(8, 3, padding="same")(X)
    X = Activation("relu")(X)
    X = BatchNormalization()(X)
    
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(X)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()

# Build model
#model = get_model(img_size, num_classes)
#model.summary()

