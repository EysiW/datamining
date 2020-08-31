import os
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, RNN, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from main.data_analysis import make_Dictionary, nof_directory, \
    extract_features_set2, make_new_Dictionary, extract_new_features, extract_features_set1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import convolutional, Dense

def model_rnn(dim):
    model = Sequential()
    model.add(Embedding(dim, 256))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model;

def run_rnn():
    # Dataset 2
    ham_corpus2, spam_corpus2 = extract_features_set2('dataset\\set2\\enron2')
    ham_corpus5, spam_corpus5 = extract_features_set2('dataset\\set2\\enron5')
    ham_corpus = pd.concat([ham_corpus2, ham_corpus5])
    spam_corpus = pd.concat([spam_corpus2, spam_corpus5])

    print(ham_corpus.shape)
    print(spam_corpus.shape)

    ham_msk = np.random.rand(len(ham_corpus)) < 0.8
    spam_msk = np.random.rand(len(spam_corpus)) < 0.8
    train_corpus = pd.concat([ham_corpus[ham_msk], spam_corpus[spam_msk]])
    test_corpus = pd.concat([ham_corpus[~ham_msk], spam_corpus[~spam_msk]])

    # Apply tfidf
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_corpus[:][0])
    test_features = vectorizer.transform(test_corpus[:][0])


    model = model_rnn(10000)
    print("Finished building model")

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history_rnn = model.fit(train_features, train_corpus[:][1], epochs=2, batch_size=1100)

    acc = history_rnn.history['acc']
    val_acc = history_rnn.history['val_acc']
    loss = history_rnn.history['loss']
    val_loss = history_rnn.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, '-', color='orange', label='training acc')
    plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, '-', color='orange', label='training acc')
    plt.plot(epochs, val_loss, '-', color='blue', label='validation acc')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

os.chdir("..")
run_rnn()

