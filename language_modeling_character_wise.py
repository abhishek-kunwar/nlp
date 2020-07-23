# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:00:55 2020

@author: abhis_000
"""

def read_data(text_file):
    with open(text_file) as f:
        data = f.read()
    return data    

#lines = read_data('../data/rhyms.txt')
def preprocessing(text_data):
    sequences = []
    splitted_data = ' '.join(text_data.split())
    for i in range(10,len(splitted_data)):
        seq = splitted_data[i-10:i+1]
        sequences.append(seq)
    return sequences

def save_sequences(seq_data,filename):
    file = open(filename,'w')
    file.write('\n'.join(seq_data))
    
lines = read_data('../data/rhyms.txt')
sequences = preprocessing(lines)
filename = 'seq_data_file'
save_sequences(sequences,filename)

def load_data(file):
    with open(file) as f:
        text = f.read()
    return text.split('\n')    
text = load_data('seq_data_file')

charac_vocab = sorted(list(set(lines)))
mapping = dict((c, i) for i, c in enumerate(charac_vocab))
vocab = len(mapping)
import numpy as np
def data_prepration(data):
    f_sequence = []
    for line in data:
        sequence = [mapping[char] for char in line]
        f_sequence.append(sequence)
    return f_sequence        
final_data = data_prepration(text) 
final_data = np.array(final_data)
X,y = final_data[:,:-1],final_data[:,-1] 
print(len(X))
print(len(y))

import keras
from keras.utils import to_categorical
from keras.layers import Dense,LSTM
from keras.models import Sequential
from pickle import load
from keras.preprocessing.sequence import pad_sequences
#from keras.utils.vis_utils import plot_model
x_final = [to_categorical(i,num_classes=vocab) for i in X]
y_final = to_categorical(y, num_classes=vocab)
x_final = np.array(x_final)

def model(x_input):
    model = Sequential()
    model.add(LSTM(units = 75,input_shape = (x_input.shape[1],x_input.shape[2])))
    model.add(Dense(vocab,activation = 'softmax'))
    model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model
    
model = model(x_final)

history = model.fit(x_final, y_final, epochs=100, verbose=2)

from pickle import dump
dump(mapping, open('mapping.pkl', 'wb'))
mapping = load(open('mapping.pkl', 'rb'))

in_text = 'Sing a son'
encoded = [mapping[char] for char in in_text]
encoded = pad_sequences([encoded], maxlen=10, truncating='pre')
encoded = to_categorical(encoded, num_classes=len(mapping))
yhat = model.predict_classes(encoded, verbose=0)
char = [key for key,value in mapping.items() if value==yhat]
    

for i in range(20):
    encoded = [mapping[char] for char in in_text]
    encoded = pad_sequences([encoded],maxlen = 10,truncating = 'pre')
    encoded = to_categorical(encoded,num_classes=len(mapping))
    yhat = model.predict_classes(encoded)
    char = [key for key,value in mapping.items() if value==yhat]
    in_text +=char[0]
    
    
    
    
    
    
    
    
    
      