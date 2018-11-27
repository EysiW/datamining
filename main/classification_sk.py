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
train_labels[nof_ham:] = 1

train_matrix = pd.read_csv('feature_enron2')

#split training and test data
msk_ham = np.random.rand(len(train_matrix[:nof_ham])) < 0.8
msk_spam = np.random.rand(len(train_matrix[nof_ham:])) < 0.8

train_data = pd.concat([train_matrix[:nof_ham][msk_ham], train_matrix[nof_ham:][msk_spam]])
test_data = pd.concat([train_matrix[:nof_ham][~msk_ham], train_matrix[nof_ham:][~msk_spam]])

# Training SVM and Naive bayes classifier

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_data.loc[:, : '2999'], train_data.loc[:, '3000'])
model2.fit(train_data.loc[:, : '2999'], train_data.loc[:, '3000'])

# Test the unseen mails for Spam

result1 = model1.predict(test_data.loc[:, : '2999'])
result2 = model2.predict(test_data.loc[:, : '2999'])
print(confusion_matrix(test_data.loc[:, '3000'], result1))
print(confusion_matrix(test_data.loc[:, '3000'], result2))
