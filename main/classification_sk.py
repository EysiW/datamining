import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from main.data import make_Dictionary
from main.data import extract_features
import pandas as pd
from data import goto

# Create a dictionary of words with its frequency
os.chdir('..')
train_dir = 'dataset\\bare\\part2'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1

train_matrix = pd.read_csv('feature_matrix')
print(type(train_matrix))

# Training SVM and Naive bayes classifier
if False:
    model1 = MultinomialNB()
    model2 = LinearSVC()
    model1.fit(train_matrix,train_labels)
    model2.fit(train_matrix,train_labels)

    # Test the unseen mails for Spam
    test_dir = 'test-mails'
    test_matrix = extract_features(test_dir)
    test_labels = np.zeros(260)
    test_labels[130:260] = 1
    result1 = model1.predict(test_matrix)
    result2 = model2.predict(test_matrix)
    print(confusion_matrix(test_labels,result1))
    print(confusion_matrix(test_labels,result2))