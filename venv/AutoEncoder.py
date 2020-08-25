# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:12:26 2020

@author: hany
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
# from functools import partial
# import logging
# import re

class AutoEncoder(BaseEstimator, TransformerMixin):
    '''
    select effective columns and transform enum column
    '''

    def __init__(self, encoding_dim):
        '''
        max_num_ratio: Proportion of numerical data
        '''
        self.encoding_dim = encoding_dim

    def fit(self, df, y=None):

        '''Shape like numerical features which is not numerical transforme into numerical;
           Features' name like channel ,code do not transforme into numerical
        '''

        self.number_cols = df.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns.tolist()
        self.number_cols = [col for col in self.number_cols if col not in ['1pd3', '1pd3', '1pd7', 'gbflag']]
        df_raw_numeric = df[self.number_cols]

        self.scaler = MinMaxScaler()
        self.scaler.fit(df_raw_numeric)
        # pickle.dump(scaler, open('scaler.pkl', 'wb'))  # pickle.load(open('scaler.pkl', 'rb'))
        x_train = pd.DataFrame(self.scaler.transform(df_raw_numeric))
        x_train = x_train.astype('float32')

        # 降低成几维:encoding_dim

        # this is our input placeholder
        input_img = Input(shape=(x_train.shape[1],))

        # encoder layers
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(10, activation='relu')(encoded)
        encoder_output = Dense(self.encoding_dim)(encoded)

        # decoder layers
        decoded = Dense(10, activation='relu')(encoder_output)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(x_train.shape[1], activation='tanh')(decoded)

        # construct the autoencoder model
        autoencoder = Model(input=input_img, output=decoded)


        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')

        # training
        autoencoder.fit(x_train, x_train,
                        epochs=100,
                        batch_size=128,
                        shuffle=True)

        # construct the encoder model for plotting
        self.encoder = Model(input=input_img, output=encoder_output)
        # encoder.save('encoder.h5')

        return self


    def transform(self, df):
        """
        """
        if not hasattr(self, 'encoder'):
            raise NotFittedError("AutoEncoder not fitted, "
                                 "call `fit` before exploiting the model.")


        df_raw_numeric = df[self.number_cols]

        x_train = pd.DataFrame(self.scaler.transform(df_raw_numeric))

        x_train = x_train.astype('float32')


        df_feature_low = self.encoder.predict(x_train)

        return df_feature_low

if __name__ == '__main__':
    Autoen = AutoEncoder(encoding_dim=3)
    data_ana_raw = pd.read_pickle(r'F:\Task\20200804_autoencoder_银联智策分析\df_enum.pkl')
    data_ana_raw_train = data_ana_raw[0:int(data_ana_raw.shape[0]*0.75)]
    data_ana_raw_test = data_ana_raw[int(data_ana_raw.shape[0]*0.75):data_ana_raw.shape[0]]

    Autoen.fit(data_ana_raw_train)

    df_result = Autoen.transform(data_ana_raw_test)
    df_result.shape