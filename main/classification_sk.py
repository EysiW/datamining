import os
from main.data_analysis import make_Dictionary, nof_directory, \
    extract_features_tdidf, make_new_Dictionary, extract_new_features
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
import matplotlib.pyplot as plt

import pandas as pd

from random import seed
#np.random.seed(2)
# Create a dictionary of words with its frequency

os.chdir('..')
train_dir = 'dataset/set1/bare/part1'
dictionary = make_Dictionary('dataset\\set2\\dictmails')

# Corpus for tf-idf
ham_corpus2, spam_corpus2 = extract_features_tdidf('dataset\\set2\\enron2')
ham_corpus5, spam_corpus5 = extract_features_tdidf('dataset\\set2\\enron5')
ham_corpus = pd.concat([ham_corpus2, ham_corpus5])
spam_corpus = pd.concat([spam_corpus2, spam_corpus5])

print(ham_corpus.shape)
print(spam_corpus.shape)

cm_dict = dict()
n = 10
result_names = ['NB', 'BNB', 'SVC', 'RF', 'KNN']
for item in result_names:
    cm_dict[item] = [[], []]

for j in range(1,n-1):
    # Split training and test
    ham_msk = np.random.rand(len(ham_corpus)) < j/n
    spam_msk = np.random.rand(len(spam_corpus)) < j/n
    train_corpus = pd.concat([ham_corpus[ham_msk], spam_corpus[spam_msk]])
    test_corpus = pd.concat([ham_corpus[~ham_msk], spam_corpus[~spam_msk]])

    # Apply tfidf
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_corpus[:][0])
    test_features = vectorizer.transform(test_corpus[:][0])

    print(train_features[1][:].shape)

    # Classifiers
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

    # Print results

    print('---------------------')
    result_list = [result_NB, result_BNB, result_SVC, result_RF, result_KNN]
    print(spam_corpus.shape)
    print(j)
    for index, value in enumerate(result_list):
        #print(result_names[i])
        cm = confusion_matrix(test_corpus[:][1], value)
        fpr = 100*(cm[0][1]/(cm[0][0] + cm[0][1]))
        fnr = 100*(cm[1][0]/(cm[1][0] + cm[1][1]))
        press = 100 - fnr - fpr
        #print('fpr: ' + str("%.2f"%fpr) + '%, fnr: ' + str("%.2f"%fnr) + '%, press: ' + str("%.2f"%press) + '%')
        #print(cm)
        #print('-------------')

        cm_dict[result_names[index]][0].append(press)
        cm_dict[result_names[index]][1].append(j/n*100)
        if j == 8:
            print(result_names[index])
            print(cm)
            print('-------------------')


plt.legend(result_names)
plt.title('Dependancy of training data')
plt.xlabel('% of data used for training')
plt.ylabel(('Precision'))
plt.show()


