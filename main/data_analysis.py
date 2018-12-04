import numpy as np
import matplotlib as mat
import os
from collections import Counter
import time
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    try:
        for mail in emails:
            with open(mail) as m:
                for i, line in enumerate(m):
                    if i == 2:  # Body of email is only 3rd line of text file
                        words = line.split()
                        all_words += words

        dictionary = Counter(all_words)
        removed_chars = list()

        for key in dictionary.keys():
            if key.isalpha() == False:
                removed_chars.append(key)
            elif len(key) == 1:
                removed_chars.append(key)
        for key in removed_chars:
            dictionary.pop(key)
        dictionary = dictionary.most_common(3000)

        return dictionary
    except UnicodeDecodeError:
        print('decodeerror')




def extract_features_tdidf(root_dir):
    mail_list = search_maildirectory(root_dir)
    ham_list = list()
    spam_list = list()
    ham = True
    try:
        for types in mail_list:
            for mails in types:
                m = open(mails, "r")
                if ham:
                    ham_list.append([m.read(), 0])
                else:
                    spam_list.append([m.read(), 1])
            ham = False
        return pd.DataFrame(ham_list), pd.DataFrame(spam_list)
    except UnicodeDecodeError:
        print('decodeerror')


def extract_features(root_dir):
    emails = search_maildirectory(root_dir)
    nof_email = emails[0].__len__() + emails[1].__len__()
    docID = 0
    dictionary = make_Dictionary('dataset/set1/bare/part1')
    features_matrix = np.zeros((nof_email,3001))
    train_labels = np.zeros(nof_email)
    for folder in emails:
        for mail in folder:
            with open(mail) as m:
                print(m)
                all_words = []
                for line in m:
                    words = line.split()
                    all_words += words
                for word in all_words:
                  wordID = 0
                  for i,d in enumerate(dictionary):
                    if d[0] == word:
                      wordID = i
                      features_matrix[docID,wordID] = all_words.count(word)
            train_labels[docID] = int(mail.split(".")[-2] == 'spam')
            features_matrix[docID,-1] = int(mail.split(".")[-2] == 'spam')
            docID = docID + 1
    return features_matrix, train_labels


def nof_directory(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    return files.__len__()


def search_maildirectory(mail_dir):
    """
    searches directory with folders including emails
    :param mail_dir:
    :return: list with name of emails
    :rtype: list()
    """
    dir = [os.path.join(mail_dir,f) for f in os.listdir(mail_dir)]
    mail_list = list()
    for folders in dir:
        files = [os.path.join(folders, f) for f in os.listdir(folders)]
        mail_list.append(files)

    return mail_list

#os.chdir('..')
#extract_features_tdidf('dataset/set2/enron5'), open('feature_enron5_tfidf', 'wb')
#file = open('feature_enrond2_tfidf', 'rb')
#print(pickle.load(file))







#print(nof_directory('dataset/set1/bare/part1'))
#extract_features('dataset/set2/enron1')
#feat = extract_features('dataset/set2/enron5')
#pd.DataFrame(feat[0]).to_csv('feature_enron5')
#pd.DataFrame(feat[1]).to_csv('test_label_enron5')
#print(search_maildirectory('dataset\\set2\\enron1')[1].__len__())
#matrix_feat = pd.read_csv('feature_enron2')
#print(matrix_feat.shape)
#for index, row in matrix_feat.iterrows():
#    print(str(index) + '.....' + str(row[3000]))