import os
from main.data_analysis import make_Dictionary, nof_directory, extract_features_tdidf
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

from random import seed
#np.random.seed(1)
# Create a dictionary of words with its frequency
os.chdir('..')
train_dir = 'dataset/set1/bare/part1'
dictionary = make_Dictionary('dataset\\set2\\dictmails')

# Corpus for tf-idf
ham_corpus2, spam_corpus2 = extract_features_tdidf('dataset\\set2\\enron2')
ham_corpus5, spam_corpus5 = extract_features_tdidf('dataset\\set2\\enron5')
ham_corpus = pd.concat([ham_corpus2, ham_corpus5])
spam_corpus = pd.concat([spam_corpus2, spam_corpus5])

# K fold cross validation


ham_msk = np.random.rand(len(ham_corpus)) < 0.7
spam_msk = np.random.rand(len(spam_corpus)) < 0.7
train_corpus = pd.concat([ham_corpus[ham_msk], spam_corpus[spam_msk]])
test_corpus = pd.concat([ham_corpus[~ham_msk], spam_corpus[~spam_msk]])
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_corpus[:][0])
test_features = vectorizer.transform(test_corpus[:][0])

print(train_features[1][:].shape)
# Prepare feature vectors per training mail and its labels
nof_ham = nof_directory('dataset/set2/enron5/ham') #-1 #adjust for number of corrupt files
nof_spam = nof_directory('dataset/set2/enron5/spam')
train_labels = np.zeros(nof_ham+nof_spam)
train_labels[nof_ham:] = 1
train_matrix = pd.read_csv('feature_enron2')

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

# Using tfidf
model_NB = MultinomialNB()
model_SVC = LinearSVC()
model_RF = RandomForestClassifier(n_estimators=100)
model_KNN = KNeighborsClassifier(n_neighbors=10)
model_BNB = BernoulliNB()

# Train Models
model_NB.fit(train_features, train_corpus[:][1])
model_SVC.fit(train_features, train_corpus[:][1])
model_RF.fit(train_features, train_corpus[:][1])
model_KNN.fit(train_features, train_corpus[:][1])
model_BNB.fit(train_features, train_corpus[:][1])

# Test the unseen mail for spam
result_NB = model_NB.predict(test_features)
result_SVC = model_SVC.predict(test_features)
result_RF = model_RF.predict(test_features)
result_KNN = model_KNN.predict(test_features)
result_BNB = model_BNB.predict(test_features)

#Print results
print(confusion_matrix(test_data.loc[:, '3000'], result1))
print(confusion_matrix(test_data.loc[:, '3000'], result2))
print('---------------------')
result_list = [result_NB, result_BNB, result_SVC, result_RF, result_KNN]
result_names = ['NB', 'GNB', 'SVC', 'RF', 'KNN']
print('total ham: ' + str(nof_ham))
print(ham_corpus.shape)

print('total spam: ' + str(nof_spam))
print(spam_corpus.shape)
for i in range(0, len(result_list)):
    print(result_names[i])
    cm = confusion_matrix(test_corpus[:][1], result_list[i])
    fpr = 100*(cm[0][1]/(cm[0][0] + cm[0][1]))
    fnr = 100*(cm[1][0]/(cm[1][0] + cm[1][1]))
    press = 100 - fnr - fpr
    print('fpr: ' + str("%.2f"%fpr) + '%, fnr: ' + str("%.2f"%fnr) + '%, press: ' + str("%.2f"%press) + '%')
    print(cm)
    print('-------------')

