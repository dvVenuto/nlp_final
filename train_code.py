import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings
import tensorflow as tf
from keras import backend as K
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer, InputSpec
# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
    
    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])
                #weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = K.sum(out,axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
      
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

import re
import nltk
nltk.download('punkt')
import os as os
import numpy as np
import itertools
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize



from keras import utils as np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

import numpy as np
from sklearn.model_selection import KFold
import time
import os
import sys
import pandas as pd

from tensorflow.python.client import device_lib

import _pickle as pickle

filename="Dropbox-Uploader/someobject.pickle"
with open(filename, 'rb') as fp:
  E = pickle.load(fp)

X_train = E.reshape(E.shape[0],68,178,50,1).astype('float32')

print(np.shape(X_train))

labeled_data = zip(E, resolved_scores)

fun_model.compile(loss='mse', optimizer='rmsprop')

y_train = resolved_scores
print(y_train)
print(np.shape(y_train))
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from softattention import Attention

from keras.models import Sequential
from keras.layers import Bidirectional, Conv1D,Input,Flatten,MaxPooling2D,TimeDistributed,LSTM,Dense, Conv2D, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.models import Model
y_scaled = preprocessing.scale(y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_scaled, test_size=0.33, random_state=42)

history = fun_model.fit(X_train, y_train, validation_split=0.33, epochs=50, batch_size=10, verbose=1)
# list all data in history
print(history.history.keys())
score = fun_model.evaluate(X_test, y_test)
print(score)

print("1D!!!!")

X_train = E.reshape(E.shape[0],68,178*50,1).astype('float32')

print(np.shape(X_train))

labeled_data = zip(E, resolved_scores)


from keras.models import Sequential
from keras.layers import Bidirectional, Conv1D,Input,Flatten,MaxPooling2D,TimeDistributed,LSTM,Dense, Conv2D, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.models import Model

cnn_input= Input(shape=(68,178*50,1))   #Frames,height,width,channel of imafe
conv1 = TimeDistributed(Conv1D(64, 3,    activation='relu'))(cnn_input)
#conv2 = TimeDistributed(Conv2D(64, (3,3), activation='relu'))(conv1)
pool1=TimeDistributed(MaxPooling1D(pool_size=4))(conv1)
att=TimeDistributed(Attention())(pool1)
flat=TimeDistributed(Flatten())(att)
#cnn_op= TimeDistributed(Dense(output_dim=3))(flat)

lstm = Bidirectional(LSTM(100, return_sequences=True, activation='tanh'))(flat)
bb = Flatten()(lstm)
op =Dense(1, activation='sigmoid')(bb)
fun_model = Model(inputs=[cnn_input], outputs=op)

fun_model.compile(loss='mse', optimizer='rmsprop')

y_train = resolved_scores
print(y_train)
print(np.shape(y_train))
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

y_scaled = preprocessing.scale(y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_scaled, test_size=0.33, random_state=42)

history = fun_model.fit(X_train, y_train, validation_split=0.33, epochs=50, batch_size=10, verbose=1)
# list all data in history
print(history.history.keys())
score = fun_model.evaluate(X_test, y_test)
