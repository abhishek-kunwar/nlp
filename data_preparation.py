# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:10:37 2020

@author: abhis_000
"""

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
text = ["The quick brown fox jumped over the lazy dog."]

countvec = CountVectorizer()
countvec.fit(text)

countvec.get_feature_names()
countvec.vocabulary_
vector = countvec.transform(text)
vector.toarray()
print(vector)
#**********************************

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
text = ["The quick brown fox jumped over the lazy dog.",
"The dog.",
"The fox"]

tfidf.fit(text)

tfidf.vocabulary_
vec = tfidf.transform(text)

vec.toarray()

#*********************************************
#data preparation using keras
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
text = np.array([['The quick brown fox jumped over the lazy dog.'],['hello hi']])

text_to_word = text_to_word_sequence(text)

text_to_word

vocab_len = len(set(text_to_word))
vocab_len

from keras.preprocessing.text import one_hot
onehot = one_hot(text, round(vocab_len*1.3))

onehot



















