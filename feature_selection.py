import csv
from random import shuffle
import string
from nltk import WordNetLemmatizer, word_tokenize
from sklearn import datasets
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
import cleaner
import stop_words_list as sw


def get_enhanced_confusion_matrix(actuals, predictions, labels):
    """"enhances confusion_matrix by adding sensivity and specificity metrics"""
    cm = confusion_matrix(actuals, predictions, labels=labels)
    sensitivity = float(cm[1][1]) / float(cm[1][0] + cm[1][1])
    specificity = float(cm[0][0]) / float(cm[0][0] + cm[0][1])
    weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
    return cm, sensitivity, specificity, weightedAccuracy


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


train_descriptions = []
test_descriptions = []
train_labels = []
test_labels = []
vectorizer = CountVectorizer(stop_words=sw.get_stop_words_list(), ngram_range=(1, 2), tokenizer=tokenize)
tfidf_transformer = TfidfTransformer(norm='l2')

data1 = csv.reader(open('/home/srinath/Downloads/datasets/combined_training.csv', 'rb'))
new_data = list(data1)
row_count = sum(1 for row in new_data)
shuffle(new_data)

for row in new_data:
    # company_description = row[4]
    company_description = cleaner.clean(str(row[4]))
    train_descriptions.append(company_description)
    NAICSid = row[2]
    train_labels.append(NAICSid)

term_freq_matrix = vectorizer.fit_transform(train_descriptions)
tf_idf_matrix = tfidf_transformer.fit_transform(term_freq_matrix)

iris = datasets.load_iris()

# data = np.array(np.arange(idx)).astype(str)
data = vectorizer.get_feature_names()


xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(tf_idf_matrix, train_labels, test_size=.25, random_state=36583)
print "building the first forest"
rf = RandomForestClassifier(n_estimators=500, min_samples_split=2, n_jobs=-1, verbose=1)
rf.fit(xTrain, yTrain)
importances = pandas.DataFrame({'name': data, 'imp': rf.feature_importances_
                                }).sort(['imp'], ascending=False).reset_index(drop=True)

cm, sensitivity, specificity, weightedAccuracy = get_enhanced_confusion_matrix(yTest, rf.predict(xTest), [0, 1])
numFeatures = len(data.count())

rfeMatrix = pandas.DataFrame({'numFeatures': [numFeatures],
                              'weightedAccuracy': [weightedAccuracy],
                              'sensitivity': [sensitivity],
                              'specificity': [specificity]})

print "running RFE on  %d features" % numFeatures

for i in range(1, numFeatures, 1):
    varsUsed = importances['name'][0:i]
    print "now using %d of %s features" % (len(varsUsed), numFeatures)
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(tf_idf_matrix[varsUsed], train_labels, test_size=.25)
    rf = RandomForestClassifier(n_estimators=500, min_samples_split=2,
                                n_jobs=-1, verbose=1)
    rf.fit(xTrain, yTrain)
    cm, sensitivity, specificity, weightedAccuracy = get_enhanced_confusion_matrix(yTest, rf.predict(xTest), [0, 1])
    print("\n" + str(cm))
    print('the sensitivity is %d percent' % (sensitivity * 100))
    print('the specificity is %d percent' % (specificity * 100))
    print('the weighted accuracy is %d percent' % (weightedAccuracy * 100))
    rfeMatrix = rfeMatrix.append(
        pandas.DataFrame({'numFeatures': [len(varsUsed)],
                          'weightedAccuracy': [weightedAccuracy],
                          'sensitivity': [sensitivity],
                          'specificity': [specificity]}), ignore_index=True)
print("\n" + str(rfeMatrix))
maxAccuracy = rfeMatrix.weightedAccuracy.max()
maxAccuracyFeatures = min(rfeMatrix.numFeatures[rfeMatrix.weightedAccuracy == maxAccuracy])
featuresUsed = importances['name'][0:maxAccuracyFeatures].tolist()

print "the final features used are %s" % featuresUsed
