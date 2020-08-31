import os
from main.data_analysis import make_Dictionary, nof_directory, \
    extract_features_set2, make_new_Dictionary, extract_new_features, extract_features_set1
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

import pandas as pd
os.chdir('..')

# Chose a seed for reproduceable results
np.random.seed(13)

# Corpus for tf-idf - Choose which dataset to use
# Dataset 1
#ham_corpus_bare, spam_corpus_bare = extract_features_set1('dataset\\set1\\bare')
#ham_corpus_stop, spam_corpus_stop = extract_features_set1('dataset\\set1\\stop')
#ham_corpus = pd.concat([ham_corpus_bare, ham_corpus_stop])
#spam_corpus = pd.concat([spam_corpus_bare, spam_corpus_stop])

# Dataset 2
ham_corpus2, spam_corpus2 = extract_features_set2('dataset\\set2\\enron2')
ham_corpus5, spam_corpus5 = extract_features_set2('dataset\\set2\\enron5')
ham_corpus = pd.concat([ham_corpus2, ham_corpus5])
spam_corpus = pd.concat([spam_corpus2, spam_corpus5])

print(ham_corpus.shape)
print(spam_corpus.shape)

cm_dict = dict()
n = 10  # Choose different sizes of test/training datasets used

result_names = ['NB', 'SVC', 'RF', 'KNN', 'BNB']
for item in result_names:
    cm_dict[item] = [[], []]

# Test with different sizes of training and test set
# In each loop, the training set increases with approx 500 mails
# and the test set decreases with the same amount
# Can be made a loop if the user want to test different size of training set
#for j in range(1, n-1):
j=8
# Split the data into training and test set randomly distributed
ham_msk = np.random.rand(len(ham_corpus)) < j/n
spam_msk = np.random.rand(len(spam_corpus)) < j/n
train_corpus = pd.concat([ham_corpus[ham_msk], spam_corpus[spam_msk]])
test_corpus = pd.concat([ham_corpus[~ham_msk], spam_corpus[~spam_msk]])

# Apply tfidf
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_corpus[:][0])
test_features = vectorizer.transform(test_corpus[:][0])

# For personal emails
gmail_corpus = extract_features_set2("dataset\\testmails\\bare")
gmail_test = vectorizer.transform(gmail_corpus[0][:][0])




print(train_features[5][:].shape)

# Classifiers
# Using default parameters, see classifiers documentations to se the values
model_NB = MultinomialNB()
model_SVC = LinearSVC()
model_RF = RandomForestClassifier(n_estimators=100)
model_KNN = KNeighborsClassifier(n_neighbors=6)
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
result_list = [result_NB, result_SVC, result_RF, result_KNN, result_BNB]
print(spam_corpus.shape)
print(j)
print(train_features)
for index, value in enumerate(result_list):
    #print(result_names[i])
    cm = confusion_matrix(test_corpus[:][1], value)
    fpr = 100*(cm[0][1]/(cm[0][0] + cm[0][1]))
    fnr = 100*(cm[1][0]/(cm[1][0] + cm[1][1]))
    acc = 100 * (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

    # Prints FPR, FNR and accuracy if comments is removed
    print('fpr: ' + str("%.2f"%fpr) + '%, fnr: ' + str("%.2f"%fnr) + '%, press: ' + str("%.2f" % acc) + '%')
    #print(cm)
    print('-------------')

    cm_dict[result_names[index]][0].append(acc)
    cm_dict[result_names[index]][1].append(j/n*100)
    if j == 8:
        print(result_names[index])
        print(cm)
        print('-------------------')

# Training and test stops here #
################################

# Print confusion matrix for each method
for method in cm_dict:
    plt.plot(cm_dict[method][1], cm_dict[method][0])
