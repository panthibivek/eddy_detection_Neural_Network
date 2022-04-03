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
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Activation, SeparableConv2D, Conv2DTranspose, Reshape, Dropout
import numpy as np
from keras.utils import np_utils
import os
import xarray as xr

def load_data_x(input_dir):
    input_file_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
        ])
    print(input_file_paths[0])
    data_x = xr.open_dataset(input_file_paths[0])
    data_x = data_x.ssh.to_numpy()
    data_x = np.float32(data_x)
    input_file_paths.pop(0)

    for abs_name in input_file_paths:
        print(abs_name)
        temp = xr.open_dataset(abs_name)
        temp = temp.ssh.to_numpy()
        temp = np.float32(temp)
        data_x = np.concatenate((data_x, temp), axis=0)
    return data_x

def load_data_y(input_dir):
    input_file_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
        ])
    print(input_file_paths[0])
    data_y = xr.open_dataset(input_file_paths[0])
    data_y = data_y.seg_mask.to_numpy()
    data_y = np.float32(data_y)
    input_file_paths.pop(0)

    for abs_name in input_file_paths:
        print(abs_name)
        temp = xr.open_dataset(abs_name)
        temp = temp.seg_mask.to_numpy()
        temp = np.float32(temp)
        data_y = np.concatenate((data_y, temp), axis=0)
    return data_y

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
        #print(y.shape)
        for i in range(self.batch_size):
            x[i] = np.expand_dims(batch_input_img[i], 2)
            y[i] = np.expand_dims(batch_target_img[i], 2)
            
        y = np_utils.to_categorical(np.reshape(y[:,:,:,0],(self.batch_size,self.img_size[0]*self.img_size[1])),3)
        return x, y

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))
    filters = 32
    # Entry block
    #x = layers.Conv2D(16, 3, strides=2, padding="same")(inputs)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation("relu")(x)

    #previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    #for filters in [(16,0.2), (16,0.3), (16,0.4)]:
    
    x1 = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("relu")(x1)

    x1 = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("relu")(x1)
    x1 = Dropout(0.2)(x1)

    p1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    
    ##------------------------------------------------------------
    x2 = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(p1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation("relu")(x2)

    x2 = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation("relu")(x2)
    x2 = Dropout(0.3)(x2)

    p2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    
    ##------------------------------------------------------------
    x3 = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(p2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation("relu")(x3)

    x3 = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation("relu")(x3)
    x3 = Dropout(0.4)(x3)

    p3 = layers.MaxPooling2D(pool_size=(2, 2))(x3)
    
    ##------------------------------------------------------------

    ### [Second half of the network: upsampling inputs] ###
    
    xc = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(p3)
    xc = layers.BatchNormalization()(xc)
    #x = Dropout(filters[1])(x)
    xc = layers.Activation("relu")(xc)
    #x = layers.BatchNormalization()(x)

    xc = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(xc)
    xc = layers.BatchNormalization()(xc)
    #x = Dropout(filters[1])(x)
    xc = layers.Activation("relu")(xc)
    xc = Dropout(0.5)(xc)
    ##------------------------------------------------------------
    up3 = concatenate([UpSampling2D((2,2))(xc), x3])
    d1 = SeparableConv2D(filters, 3, padding="same", use_bias=False)(up3) 
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    
    d1 = SeparableConv2D(filters, 3, padding="same", use_bias=False)(d1) 
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Dropout(0.4)(d1)
    ##------------------------------------------------------------
    
    up2 = concatenate([UpSampling2D((2,2))(d1), x2])
    d2 = SeparableConv2D(filters, 3, padding="same", use_bias=False)(up2) 
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    
    d2 = SeparableConv2D(filters, 3, padding="same", use_bias=False)(d2) 
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Dropout(0.3)(d2)
    ##------------------------------------------------------------
    
    up3 = concatenate([UpSampling2D((2,2))(d2), x1])
    d3 = SeparableConv2D(filters, 3, padding="same", use_bias=False)(up3) 
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    
    d3 = SeparableConv2D(filters, 3, padding="same", use_bias=False)(d3) 
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Dropout(0.2)(d3)
    ##------------------------------------------------------------

    # Add a per-pixel classification layer
    
    X = SeparableConv2D(num_classes, (1,1), padding="same", use_bias=False)(d3)   
    X = Reshape((img_size[0] * img_size[1], num_classes))(X) 
    outputs = Activation("softmax")(X)
    #outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(X)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()

# Build model
#model = get_model(img_size, num_classes)
#model.summary()

