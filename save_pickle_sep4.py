#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
import pickle
from random import shuffle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.metrics import accuracy_score
from sklearn import metrics
import cleaner
import stop_words_list as sw
from sklearn import cross_validation
import sys
from sklearn import svm
import sklearn.feature_selection


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def tokenize(text):
    txt = "".join([ch for ch in text if ch not in string.punctuation])
    # print txt
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in word_tokenize(txt)]


# vectorizer = CountVectorizer(stop_words=sw.get_stop_words_list(), ngram_range=(1, 2), tokenizer=tokenize,max_features=25000)
vectorizer = CountVectorizer(stop_words=sw.get_stop_words_list(), ngram_range=(1, 2), tokenizer=tokenize)
tfidf_transformer = TfidfTransformer(norm='l2')
# nb_classifier = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-3, n_iter=10, random_state=42, shuffle=True)
#nb_classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, shuffle=True, n_jobs=-1)
nb_classifier = Pipeline([('classification', SelectFromModel(SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, shuffle=True, n_jobs=-1))),('feature_selection', RandomForestClassifier())])
# nb_classifier = SGDClassifier(loss='log', penalty='l2', alpha=1e-3,shuffle=True)
# nb_classifier = svm.SVC(kernel='linear', C=1.0)
# nb_classifier = MultinomialNB()

train_descriptions = []
test_descriptions = []
train_labels = []
test_labels = []
naics_codes = {}


def use_all_training_data(new_data):
    rcount = 0
    for row in new_data:
        rcount += 1
        if rcount % 1000 == 0:
            print rcount
        # company_description = row[4]
        company_description = cleaner.clean(str(row[4]))
        train_descriptions.append(company_description)
        NAICSid = row[2]
        train_labels.append(NAICSid)
    return train_descriptions, train_labels

def get_naics_description(naics_codes_file):
    for row in naics_codes_file:
        key = row[0]
        value = row[1]
        naics_codes[key] = value


def split_training_data(new_data, row_count):
    row_index = 0
    for row in new_data:

        if row_index < 0.8 * row_count:

            if row_index % 1000 == 0:
                print row_index

            company_description = row[10]
            #company_description = cleaner.clean(row[10])
            # train_descriptions.append(str(company_description))
            train_descriptions.append(str(company_description))
            naicsid = row[6]
            train_labels.append(naicsid)
        elif row_index >= 0.8 * row_count:
            company_description = row[10]
            #company_description = cleaner.clean(str(row[10]))
            test_descriptions.append(company_description)
            naicsid = row[6]
            test_labels.append(naicsid)
        row_index += 1

    ''''y = open('Training.csv', 'w')
    z = csv.writer(y)
    leng= len(train_labels)
    print leng
    i=0
    while(i < leng):
        data = [[train_labels[i].encode('ascii','ignore'),train_descriptions[i].encode('ascii','ignore')]]
        print data

        z.writerows(data)
        i=i+1
        print i
    a = open('Test.csv', 'w')
    b = csv.writer(a)
    leng= len(test_labels)
    print leng
    i=0
    while(i < leng):
        data = [[test_labels[i].encode('ascii','ignore'),test_descriptions[i].encode('ascii','ignore')]]
        print data

        b.writerows(data)
        i=i+1
        print i'''




def cross_validate_data(new_data, row_count):
    use_all_training_data(new_data)
    print len(train_descriptions)
    term_freq_matrix = vectorizer.fit_transform(train_descriptions)
    tf_idf_matrix = tfidf_transformer.fit_transform(term_freq_matrix)
    # nb_classifier.fit(tf_idf_matrix,train_labels)
    print type(tf_idf_matrix)
    print type(train_labels)
    scores = cross_validation.cross_val_score(nb_classifier, tf_idf_matrix, np.asarray(train_labels), cv=11)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def fit_vocabulary():
    term_freq_matrix = vectorizer.fit_transform(train_descriptions)
    tf_idf_matrix = tfidf_transformer.fit_transform(term_freq_matrix)
    for aa in vectorizer.get_feature_names():
        print aa.encode('utf-8')
    return (term_freq_matrix, tf_idf_matrix)


def create_and_save_classifier():

    term_freq_matrix, tf_idf_matrix = fit_vocabulary()
    # nb_classifier = MultinomialNB().fit(tf_idf_matrix,train_labels)
    nb_classifier.fit(tf_idf_matrix, train_labels)
    clf = Pipeline([('feature_selection', SelectFromModel(SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, shuffle=True, n_jobs=-1))),('classification', RandomForestClassifier())])
    #clf.fit(tf_idf_matrix, train_labels)

    '''
    f = open('model_sep4.pickle', 'wb')
    pickle.dump(nb_classifier, f)
    f.close()
    '''


def check_test_data():
    file = open("testing_data_sep3.txt", "w")
    x_new_counts = vectorizer.transform(test_descriptions)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)

    predicted = nb_classifier.predict(x_new_tfidf)
    count = 0
    temp = str("=>")
    delimit = str("\n")
    co = str(" | ")

    file.write("Doc+temp+Predicted+temp+Target"+delimit)

    for doc, category in zip(test_descriptions, predicted):
        #print('%r => %s , %s' % (doc, category, test_labels[count]))
        file.write(doc+temp+co+category+temp+co+test_labels[count]+delimit)
        count += 1

    print np.mean(predicted == test_labels)
    print "----------"
    # print accuracy_score(test_labels, predicted)
    print(metrics.classification_report(test_labels, predicted))
    file.close()
'''

def check_test_data():

    vocab_file = open('vocab_sep6.pickle')
    vec = pickle.load(vocab_file)
    print "******************************"

    idf_file = open('idf_sep6.pickle')
    idf_transformer = pickle.load(idf_file)

    f = open('model_sep6.pickle')
    classifier = pickle.load(f)

    f = open('codes_sep6.pickle')
    naics_codes_desc = pickle.load(f)

    x_new_counts = vec.transform(test_descriptions)
    x_new_tfidf = idf_transformer.transform(x_new_counts)
    predicted = classifier.predict(x_new_tfidf)


    print np.mean(predicted == test_labels)
    print "----------"
    # print accuracy_score(test_labels, predicted)
    print(metrics.classification_report(test_labels, predicted))
    file.close()
'''

if __name__ == '__main__':

    data = csv.reader(open('combined_training_sep3 (copy).csv', 'rb'))
    new_data = list(data)
    print new_data
    row_count = sum(1 for row in new_data)
    shuffle(new_data)

    #naics_codes_file = csv.reader(open('naics_codes.csv', 'rb'))
    #get_naics_description(naics_codes_file)

    split_training_data(new_data, row_count)
    create_and_save_classifier()
    check_test_data()
    #print "DONE"

    #cross_validate_data(new_data,row_count)