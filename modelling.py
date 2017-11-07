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
import random
import sys
from sklearn import svm
import sklearn.feature_selection
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


def tokenize(text):
    txt = "".join([ch for ch in text if ch not in string.punctuation])
    # print txt
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in word_tokenize(txt)]


# vectorizer = CountVectorizer(stop_words=sw.get_stop_words_list(), ngram_range=(1, 2), tokenizer=tokenize,max_features=25000)
vectorizer = CountVectorizer(stop_words=sw.get_stop_words_list(),ngram_range=(1, 2), tokenizer=tokenize)
tfidf_transformer = TfidfTransformer(norm='l2')
# nb_classifier = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-3, n_iter=10, random_state=42, shuffle=True)
#nb_classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, shuffle=True, n_jobs=-1)
nb_classifier = Pipeline([('classification', SelectFromModel(SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, shuffle=True, n_jobs=-1))),('feature_selection', RandomForestClassifier(n_estimators=1500, min_samples_split=2, n_jobs=-1, verbose=1))])
#nb_classifier = SGDClassifier(loss='log', penalty='l2', alpha=1e-3,shuffle=True)
#nb_classifier = svm.SVC(kernel='linear', C=1.0)
# nb_classifier = MultinomialNB()
#estimator = SVR(kernel="linear")
#nb_classifier = RFE(estimator, 5, step=1)

train_descriptions = []
test_descriptions = []
train_labels = []
test_labels = []
naics_codes = {}
company=[]




'''def use_all_training_data(new_data):
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
'''
def get_naics_description(naics_codes_file):
    for row in naics_codes_file:
        key = row[0]
        value = row[1]
        naics_codes[key] = value


def split_training_data(new_data, row_count):
    row_index = 0
    for row in new_data:
        #print row
        if row_index < 0.8 * row_count:

            if row_index % 1000 == 0:
                print row_index
            if(a==0):
                company_description = row[10]

            #company_description = row[4]
            if(a==1):
                company_description = cleaner.clean(row[10])
            # train_descriptions.append(str(company_description))
            train_descriptions.append(str(company_description))
            naicsid = row[6]


            #naicsid = row[2]
            train_labels.append(naicsid)
        elif row_index >= 0.8 * row_count:
            if(a==0):
                company_description = row[10]
            #company_description = row[4]
            if(a==1):
                company_description = cleaner.clean(str(row[10]))
            test_descriptions.append(company_description)
            naicsid = row[6]
            #naicsid = row[2]
            company.append(row[0])
            test_labels.append(naicsid)
        row_index += 1



def fit_vocabulary():
    file1 = open("features.txt", "w")
    i=1
    feature=[]
    term_freq_matrix = vectorizer.fit_transform(train_descriptions)
    tf_idf_matrix = tfidf_transformer.fit_transform(term_freq_matrix)
    for aa in vectorizer.get_feature_names():
        print i
        print aa.encode('utf-8')
        feature.append(aa.encode('utf-8'))
        i=i+1
    leng=len(feature)
    print leng
    '''Fileopener= open('feat.csv', 'w')
    Filewriter = csv.writer(Fileopener)

    while(i < leng):
        data=[[i+1,feature[i].encode('ascii','ignore')]]
        Filewriter .writerows(data)


    file1.close()'''
    return (term_freq_matrix, tf_idf_matrix)



def create_and_save_classifier():

    term_freq_matrix, tf_idf_matrix = fit_vocabulary()
    # nb_classifier = MultinomialNB().fit(tf_idf_matrix,train_labels)
    nb_classifier.fit(tf_idf_matrix, train_labels)
    #clf = Pipeline([('feature_selection', SelectFromModel(SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, shuffle=True, n_jobs=-1))),('classification', RandomForestClassifier())])
    #clf.fit(tf_idf_matrix, train_labels)


'''def cross_validate_data(new_data, row_count):
    use_all_training_data(new_data)
    print len(train_descriptions)
    term_freq_matrix = vectorizer.fit_transform(train_descriptions)
    tf_idf_matrix = tfidf_transformer.fit_transform(term_freq_matrix)
    # nb_classifier.fit(tf_idf_matrix,train_labels)
    print type(tf_idf_matrix)
    print type(train_labels)
    scores = cross_validation.cross_val_score(nb_classifier, tf_idf_matrix, np.asarray(train_labels), cv=3)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))'''



def check_test_data():
    file = open("testing_data.txt", "w")
    x_new_counts = vectorizer.transform(test_descriptions)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)

    predicted = nb_classifier.predict(x_new_tfidf)
    count = 0
    temp = str("|")
    delimit = str("\n")
    co = str(" | ")
    docu=[]
    pre=[]
    targ=[]

    file.write("Doc+temp+Predicted+temp+Target"+delimit)
    for doc, category in zip(test_descriptions, predicted):
        #print('%r => %s , %s' % (doc, category, test_labels[count]))
        file.write(doc+temp+co+category+temp+co+test_labels[count]+delimit)

        docu.append(doc)
        pre.append(category)
        targ.append(test_labels[count])
        count += 1

    print type(doc),type(category),type(co)
    print np.mean(predicted == test_labels)
    print "----------"
    # print accuracy_score(test_labels, predicted)
    print(metrics.classification_report(test_labels, predicted))

    if (a==0):
        leng=len(docu)
        print leng
        print type(company)
        print len(company)
        i=0
        Fileopener = open('data_validation0.csv', 'w')
        Filewriter = csv.writer(Fileopener)
        while(i < leng):
            print "PRINTING"
            #print docu[i],pre[i],targ[i]
            try:
                data=[[company[i],docu[i].encode('ascii','ignore'),pre[i].encode('ascii','ignore'),targ[i].encode('ascii','ignore')]]
            except Exception,e:
                print "Exception"
            Filewriter.writerows(data)
            i=i+1
            print data
            print i
    if (a==1):
        leng=len(docu)
        print leng
        i=0
        Fileopener = open('data_validation1.csv', 'w')
        Filewriter = csv.writer(Fileopener)
        while(i < leng):
            print "PRINTING"
            #print docu[i],pre[i],targ[i]
            try:
                data=[[docu[i].encode('ascii','ignore'),pre[i].encode('ascii','ignore'),targ[i].encode('ascii','ignore')]]
            except Exception,e:
                print "Exception"
            Filewriter.writerows(data)
            i=i+1
            print data
            print i
    print "DONE"



    file.close()



if __name__ == '__main__':

    data = csv.reader(open('Data.csv', 'rb'))
    #data = csv.reader(open('combined_training_sep3.csv', 'rb'))
    new_data = list(data)
    #print new_data
    row_count = sum(1 for row in new_data)
    shuffle(new_data)
    #random.seed(new_data)
    a=0
    split_training_data(new_data, row_count)
    create_and_save_classifier()
    check_test_data()
    #cross_validate_data(new_data,row_count)
    #print "DONE"
    a=1

    split_training_data(new_data, row_count)
    create_and_save_classifier()
    check_test_data()