# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:25:04 2022

@author: Jip de Kok

Credits for this code go to the original authors of the cancerAI
integrativeVAEs Github repository
(https://github.com/CancerAI-CL/IntegrativeVAEs)
"""

from keras import regularizers
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model, clone_model
from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedKFold
from metrics import calculate_metrics
import pickle
import hdbscan
import time

import argparse
import os
import tensorflow as tf
import random
from keras import backend as K
from tensorflow.keras import optimizers
from keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout
from keras.models import Model

from scripts.common import sse, bce, mmd, sampling, kl_regu
from keras.losses import mean_squared_error,binary_crossentropy
import numpy as np
import pandas as pd
from tensorflow.random import set_seed

set_seed(5192)
os.environ['PYTHONHASHSEED'] = str(5192)
np.random.seed(5192)
random.seed(5192)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


class xvae():
    # SET 1 SHOULD ALWAYS BE NUMERICAL AND SET 2 BINARY (CATEGORICAL)
    
    def __init__(self, s1_input_size, s2_input_size, ds1 = 48, ds2 = None,
                 ds12 = None, ls = 32, weighted = True,
                 act = "elu", dropout = 0.2, distance = "kl", beta = 25,
                 epochs = 250, bs = 64, save_model = False):
        
        '''
        
        Parameters
        ----------
        s1_input_size : integer
            Number of features in first data set.
        s2_input_size : integer
            Number of features in second data set.
        ds1 : integer
            The intermediate dense layers size. if ds2 and ds12 are no
            specified, all dense layers will be size of ds1. Otherwise,
            Only the first dense layer of data set 1 will be set to ds1
        ds2 : integer, optional
            The first intermediate dense layer size of the of the
            second data set. If None, will be set to ds1. The default is None.
        ds12 : integer, optional
            The second intermediate dense layer size that combines the first
            and second data set. If None, will be set to ds1. The default is None.
        ls : TYPE
            latent dimension size
        act : TYPE
            activation function
            DESCRIPTION. The default is "elu".
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.2.
        distance : TYPE, optional
            DESCRIPTION. The default is "kl".
        beta : TYPE, optional
            DESCRIPTION. The default is 25.
        epochs : TYPE, optional
            DESCRIPTION. The default is 250.
        bs : TYPE, optional
            DESCRIPTION. The default is 64.
        save_model : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        
        # Initiate components of class object
        self.args = argparse.ArgumentParser()
        self.vae = None
        self.encoder = None
        
        # Fil args component of class object
        self.args.s1_input_size = s1_input_size
        self.args.s2_input_size = s2_input_size
        self.args.ds1 = ds1
        if ds2 == None:
            self.args.ds2 = ds1
        else:
            self.args.ds2 = ds2
        if ds12 == None:
            self.args.ds12 = ds1
        else:
            self.args.ds12 = ds12
        self.args.ls = ls
        self.args.act = act
        self.args.dropout = dropout
        self.args.distance = distance
        self.args.beta = beta
        self.args.epochs = epochs
        self.args.bs = bs
        self.args.weighted = weighted
        self.args.save_model = save_model
        
        
    def build_model(self):
        tf.random.set_seed(5192)
        os.environ['PYTHONHASHSEED'] = str(5192)
        np.random.seed(5192)
        random.seed(5192)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # Build the encoder network
        # ------------ Input -----------------
        s1_inp = Input(shape=(self.args.s1_input_size,))
        s2_inp = Input(shape=(self.args.s2_input_size,))
        inputs = [s1_inp, s2_inp]
        


        # ------------ Concat Layer -----------------
        # First hidden layer for set1
        x1 = Dense(self.args.ds1, activation=self.args.act)(s1_inp)
        x1 = BN()(x1)

        # First hidden layer for set2
        x2 = Dense(self.args.ds2, activation=self.args.act)(s2_inp)
        x2 = BN()(x2)
        
        # Combine first two hidden layers
        x = Concatenate(axis=-1)([x1, x2])

        # Second hidden layers with first two hidden layers as input
        x = Dense(self.args.ds12, activation=self.args.act)(x)
        x = BN()(x)

        # ------------ Embedding Layer --------------
        # Encoding layer
        z_mean = Dense(self.args.ls, name='z_mean')(x)
        z_log_sigma = Dense(self.args.ls, name='z_log_sigma', kernel_initializer='zeros')(x)
        z = Lambda(sampling, output_shape=(self.args.ls,),
                   name='z')([z_mean, z_log_sigma])

        self.encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.Z_encoder = Model(inputs, z_mean, name='Z_encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        latent_inputs = Input(shape=(self.args.ls,), name='z_sampling')
        x = latent_inputs
        x = Dense(self.args.ds12, activation=self.args.act)(x)
        x = BN()(x)
        
        x=Dropout(self.args.dropout)(x)
        # ------------ Dense branches ------------
        x1 = Dense(self.args.ds1, activation=self.args.act)(x)
        x1 = BN()(x1)
        x2 = Dense(self.args.ds2, activation=self.args.act)(x)
        x2 = BN()(x2)

        # ------------ Out -----------------------
        s1_out = Dense(self.args.s1_input_size, activation='linear')(x1) # Manual change here
        s2_out = Dense(self.args.s2_input_size, activation='sigmoid')(x2) # Manual change here

        decoder = Model(latent_inputs, [s1_out, s2_out], name='decoder')
        decoder.summary()
        
        outputs = decoder(self.encoder(inputs)[2])
        self.decoder = decoder
        self.vae = Model(inputs, outputs, name='vae_x')

        if self.args.distance == "mmd":
            true_samples = K.random_normal(K.stack([self.args.bs, self.args.ls]))
            distance = mmd(true_samples, z)
        elif self.args.distance == "kl":
            distance = kl_regu(z_mean,z_log_sigma)
        else:
            raise ValueError(f"{self.distance} not recognised as distance.")
        
        if self.args.weighted:
            s1_loss = mean_squared_error(inputs[0], outputs[0]) * self.args.s1_input_size # Manual change here
            s2_loss = binary_crossentropy(inputs[1], outputs[1]) * self.args.s2_input_size # Manual change here
        else:
            s1_loss = mean_squared_error(inputs[0], outputs[0]) # Manual change here
            s2_loss = binary_crossentropy(inputs[1], outputs[1]) # Manual change here
        
        
        self.s1_loss = s1_loss
        self.s2_loss = s2_loss
        
        if self.args.weighted:
            reconstruction_loss = (s1_loss+s2_loss)/(self.args.s1_input_size + self.args.s2_input_size)
        else:
            reconstruction_loss = s1_loss+s2_loss
            
        vae_loss = K.mean(reconstruction_loss + self.args.beta * distance)
        self.vae.add_loss(vae_loss)
        
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                               epsilon=None, amsgrad=False, decay=0.001)
        self.vae.compile(optimizer=adam,
                         metrics=[binary_crossentropy])
        
        self.input = inputs
        self.output = z_mean
        
        self.reconstruction = outputs

        
        self.vae.summary()
        
        
    def train(self, s1_train, s2_train, s1_test, s2_test):
        tf.random.set_seed(5192)
        os.environ['PYTHONHASHSEED'] = str(5192)
        np.random.seed(5192)
        random.seed(5192)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        self.vae.fit([s1_train, s2_train], epochs=self.args.epochs,
                     batch_size=self.args.bs, shuffle=True,
                     validation_data=([s1_test, s2_test], None))
        if self.args.save_model:
            self.save_model()

    def predict(self, s1_data, s2_data, output = 'encoder'):
        '''
        runs the trained xvae on new data to encode or reproduce the data.

        Parameters
        ----------
        s1_data : ndarray
            2-dimensional array of the first input set. Rows are samples and
            columns are variables.
        s2_data : ndarray
            2-dimensional array of the second input set. Rows are samples and
            columns are variables.
        output : str, optional
            String specifying what output to return. Can be set to "encoder" to
            only run the encoder, and return the latent feautures. Or to
            "decoder" to run the entire xvae and return recreated input
            variables. The default is 'encoder'.

        Returns
        -------
        ndarray
            2-dimensional array of the encoded latent features or recreated
            input variables, depending on what output is requested. Rows are
            samples and columns are variables.

        '''
        tf.random.set_seed(5192)
        os.environ['PYTHONHASHSEED'] = str(5192)
        np.random.seed(5192)
        random.seed(5192)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        if output =="encoder":
            return self.encoder.predict([s1_data, s2_data],
                                        batch_size=self.args.bs)[0]
        elif output =="decoder":
            return self.decoder.predict(self.encoder.predict(
                [s1_data, s2_data], batch_size=self.args.bs)[0])
    
    def save_model(self, path = "xvae_model", force_path = True):
        '''
        Locally save xvae encoder

        Parameters
        ----------
        path : str, optional
            Directory where the  xvae model should be stored locally. The
            default is "xvae_model".
        force_path : boolean, optional
            Inidicates whether the directory should be created if non-existent.
            The default is True.

        Returns
        -------
        None.

        '''
        
        # Create directory if non-existent
        if force_path:
            if not os.path.exists(path):
                os.makedirs(path)
            
        # Save model architecture
        architecture = pd.DataFrame(index = ["s1_input_size", "s2_input_size",
                                             "ds1", "ds2", "ds12", "ls",
                                             "weighted", "act" , "dropout",
                                             "distance", "beta", "epochs",
                                             "bs", "save_model"],
                                    columns = ["value"])
        for i in architecture.index:
            architecture.loc[i,"value"] = getattr(self.args, i)
        architecture.to_csv(f"{path}/model_architecture.csv")
        
        # Save model weights
        self.vae.save_weights(f"{path}/model_weights")
        
        return


def load_xvae_model(path):
    '''
    Load a saved xvae encoder.

    Parameters
    ----------
    path : str
        Directory where the saved xvae model is located.

    Returns
    -------
    model : xvae
        xvae encoder.

    '''
    # Load model architecture
    architecture = pd.read_csv(f"{path}/model_architecture.csv", index_col = 0)
    
    # Initiate xvae with appropriate architecture
    model = xvae(s1_input_size = int(architecture.loc["s1_input_size","value"]),
                 s2_input_size = int(architecture.loc["s2_input_size","value"]),
                 beta = int(architecture.loc["beta","value"]),
                 ds1 = int(architecture.loc["ds1","value"]),
                 ds2 = int(architecture.loc["ds2","value"]),
                 ds12 = int(architecture.loc["ds12","value"]),
                 ls = int(architecture.loc["ls","value"]),
                 dropout=float(architecture.loc["dropout","value"]),
                 bs = int(architecture.loc["bs","value"]),
                 distance = architecture.loc["distance","value"])

    model.build_model()
    
    # Load and apply model weights
    model.vae.load_weights(f"{path}/model_weights")
    
    return model
