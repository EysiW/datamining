import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from main.data import make_Dictionary
from main.data import extract_features
from data import nof_directory
import pandas as pd
from data import goto

# Create a dictionary of words with its frequency
os.chdir('..')
train_dir = 'dataset/set1/bare/part1'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels
nof_ham = nof_directory('dataset/set2/enron2/ham')
nof_spam = nof_directory('dataset/set2/enron2/spam')
train_labels = np.zeros(nof_ham+nof_spam)
train_labels[:nof_ham] = 1

train_matrix = pd.read_csv('feature_enron2')


# Training SVM and Naive bayes classifier

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