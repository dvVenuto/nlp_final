import data_utils
from keras import utils as np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

import numpy as np
from sklearn.model_selection import KFold
from qwk import quadratic_weighted_kappa
import time
import os
import sys
import pandas as pd

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

is_regression = False
embedding_size=50
num_tokens=6
essay_set_id = 1

# print flags info

# hyper-parameters end here
training_path = 'training_set_rel3.tsv'
essay_list, resolved_scores, essay_id = data_utils.load_training_data(training_path, essay_set_id)

max_score = max(resolved_scores)
min_score = min(resolved_scores)
if essay_set_id == 7:
    min_score, max_score = 0, 30
elif essay_set_id == 8:
    min_score, max_score = 0, 60

#print 'max_score is {} \t min_score is {}\n'.format(max_score, min_score)


# include max score
score_range = range(min_score, max_score+1)

word_idx, word2vec = data_utils.load_glove(num_tokens, dim=embedding_size)

#word_idx, _ = data_utils.build_vocab(essay_list, vocab_limit)

# load glove
                            

vocab_size = len(word_idx) + 1
# stat info on data set
#essay_list=essay_list[1:2]

max_sent_size=0
for essay in essay_list:
    for w in essay:
        leng=len(w)
        print(leng)
        if leng > max_sent_size:
            max_sent_size=leng		

print(resolved_scores)

print(max_sent_size)
#print(essay_list)
#print 'The length of score range is {}'.format(len(score_range))

E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size,word2vec,embedding_size)
print(len(resolved_scores))

import pickle
import cPickle
with open(r"someobject.pickle", "wb") as output_file:
	cPickle.dump(E, output_file)

#pickle.dump( E, open( "save.p", "wb" ) )

X_train = E.reshape(E.shape[0],68,178,50,1).astype('float32')

print(np.shape(X_train))

labeled_data = zip(E, resolved_scores)


from keras.models import Sequential
from keras.layers import Bidirectional, Conv1D,Input,Flatten,MaxPooling2D,TimeDistributed,LSTM,Dense, Conv2D, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.models import Model
from softattention import Attention

cnn_input= Input(shape=(68,178,50,1))   #Frames,height,width,channel of imafe
conv1 = TimeDistributed(Conv2D(100, (3,3),    activation='relu'))(cnn_input)
#conv2 = TimeDistributed(Conv2D(64, (3,3), activation='relu'))(conv1)
pool1=TimeDistributed(TimeDistributed(Attention()))(conv1)
flat=TimeDistributed(Flatten())(pool1)
#cnn_op= TimeDistributed(Dense(output_dim=3))(flat)

lstm = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(flat)
bb = Flatten()(lstm)
op =Dense(1, activation='sigmoid')(bb)
fun_model = Model(inputs=[cnn_input], outputs=op)

from keras.utils.np_utils import to_categorical

#model = Sequential()
#model.add(Dropout(0.5,input_shape=(178,50,1)))
#model.add(TimeDistributed(Conv2D(64, kernel_size=13, activation='relu')))
#model.add(TimeDistributed(GlobalAveragePooling1D()))
#model.add(LSTM())
#model.add(GlobalAveragePooling1D())
#model.add(Dense(10, activation='softmax'))
#model.compile(loss='mse', optimizer='rmsprop')

fun_model.compile(loss='mse', optimizer='rmsprop')

y_train = resolved_scores
print(y_train)
print(np.shape(y_train))
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from softattention import Attention

y_scaled = preprocessing.scale(y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_scaled, test_size=0.33, random_state=42)

history = fun_model.fit(X_train, y_train, validation_split=0.33, epochs=1, batch_size=10, verbose=1)
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
    
history = fun_model.fit(X_train, y_train, validation_split=0.33, epochs=1, batch_size=10, verbose=1)
# list all data in history
print(history.history.keys())
score = fun_model.evaluate(X_test, y_test)
