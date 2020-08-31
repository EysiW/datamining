import os
from main.data_analysis import make_Dictionary, nof_directory, \
    extract_features_set2, make_new_Dictionary, extract_new_features
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import KFold
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.cluster.k_means_ import k_means
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import layers

import pandas as pd

# Corpus
ham_corpus2, spam_corpus2 = extract_features_set2('dataset\\set2\\enron2')
ham_corpus5, spam_corpus5 = extract_features_set2('dataset\\set2\\enron5')
corpus = pd.concat([ham_corpus2, ham_corpus5, spam_corpus2, spam_corpus5])


# Prepare feature vectors per training mail and its labels
train_matrix = extract_new_features(train_corpus, dictionary)
print(train_matrix)
# Split training and test data, partition 4:1
msk_ham = np.random.rand(len(train_matrix[:nof_ham])) < 0.8
msk_spam = np.random.rand(len(train_matrix[nof_ham:])) < 0.8
train_data = pd.concat([train_matrix[:nof_ham][msk_ham], train_matrix[nof_ham:][msk_spam]])
test_data = pd.concat([train_matrix[:nof_ham][~msk_ham], train_matrix[nof_ham:][~msk_spam]])


# Training SVM and Naive bayes classifier using dictionary
model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_data.loc[:, : '2999'], train_data.loc[:, '3000'])
model2.fit(train_data.loc[:, : '2999'], train_data.loc[:, '3000'])

# Test the unseen mails for Spam
result1 = model1.predict(test_data.loc[:, : '2999'])
result2 = model2.predict(test_data.loc[:, : '2999'])

print(confusion_matrix(test_data.loc[:, '3000'], result1))
print(confusion_matrix(test_data.loc[:, '3000'], result2))