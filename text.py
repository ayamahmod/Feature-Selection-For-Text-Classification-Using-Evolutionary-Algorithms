import os
import sys
import io
from time import time
from sklearn.utils.extmath import density
import argparse
from collections import defaultdict
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
# import nltk
# from nltk.corpus import stopwords
# from nltk.classify.scikitlearn import SklearnClassifier


ten_most = ['earn', 'acq',
            'money-fx', 'grain',
            'crude', 'trade',
            'interest', 'ship',
            'wheat', 'corn']

k_features = [500, 1000, 2000, 5000]


def read_data(d):
    categories = os.listdir(d)
    data = []
    for c in categories:
        full_path = os.path.join(d, c)
        files = [os.path.join(full_path, f)
                 for f in os.listdir(full_path)]

        for filename in files:
            with io.open(filename, 'r') as f:
                text = f.read()
                data.append((c, text))
    return data


def build_tfidf(train_data, test_data):
    counter = CountVectorizer(stop_words='english')
    raw = [data[1] for data in train_data]
    train_tf = counter.fit_transform(raw)
    raw = [data[1] for data in test_data]
    test_tf = counter.transform(raw)

    transformer = TfidfTransformer()
    train_tfidf = transformer.fit_transform(train_tf)
    test_tfidf = transformer.transform(test_tf)
    return train_tfidf, test_tfidf

def select_features(train_X, train_y, test_X, k):
    selector = SelectKBest(chi2, k=k)
    selector.fit(train_X, train_y)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X

def get_word_vector(text):
    stop = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.lower() not in stop]
    vector = nltk.FreqDist(tokens)
    vector = {k: v for k, v in vector.iteritems() if v >= 3}
    return vector


def train(X, y):
    clf = svm.SVC()
    clf.fit(X, y)
    return clf


def evaluate(labels, true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    fscore = []

    TP = 0.0
    TPFP = 0.0
    TPFN = 0.0
    for label in labels:
        tp = predicted_labels[true_labels == label]
        tp = np.sum(tp == label)

        tpfp = np.sum(predicted_labels == label)
        if tpfp == 0:
            precision = 1
        else:
            precision = float(tp) / tpfp

        tpfn = np.sum(true_labels == label)
        if tpfn == 0:
            recall = 1
        else:
            recall = float(tp) / tpfn
        fscore.append(2*(precision*recall)/(precision+recall))

        TP += tp
        TPFP += tpfp
        TPFN += tpfn
    if TPFP == 0:
        p = 1
    else:
        p = TP/TPFP
    if TPFN == 0:
        r = 1
    else:
        r = TP/TPFN
    # microaverage = 2*(p*r)/(p+r)
    average_score = metrics.f1_score(true_labels, predicted_labels)
    return fscore, average_score


def benchmark(clf, train_X, train_y, test_X, test_y):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(train_X, train_y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_X)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(test_y, pred, average='micro')
    print("f1-score:   %0.3f" % score)

    # scores = metrics.f1_score(test_y, pred, average=None)
    # print(scores)

    # print("confusion matrix:")
    # print(metrics.confusion_matrix(test_y, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help="the train data directory")
    parser.add_argument('test', help="the test data directory")

    args = parser.parse_args()

    print('read data')
    train_data = read_data(args.train)
    test_data = read_data(args.test)

    train_labels = [data[0] for data in train_data]
    test_labels = [data[0] for data in test_data]
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_labels)
    test_y = encoder.transform(test_labels)
    ten_encoded = encoder.transform(ten_most)
    print("build tfidf")
    train_tfidf, test_tfidf = build_tfidf(train_data, test_data)

    print("training")
    for k in k_features:
        print("#"*80)
        print("select {0} features".format(k))
        train_X, test_X = select_features(train_tfidf, train_labels,
                                          test_tfidf, k)
        clf = svm.LinearSVC()
        benchmark(clf, train_X, train_y, test_X, test_y)

        for degree in range(1, 6):
            clf = svm.SVC(kernel='poly', degree=degree)
            benchmark(clf, train_X, train_y, test_X, test_y)

        for gamma in [0.6, 0.8, 1, 1.2]:
            clf = svm.SVC(kernel='rbf', gamma=gamma)
            benchmark(clf, train_X, train_y, test_X, test_y)

        clf = KNeighborsClassifier()
        benchmark(clf, train_X, train_y, test_X, test_y)

        clf = NearestCentroid()
        benchmark(clf, train_X, train_y, test_X, test_y)

        clf = MultinomialNB()
        benchmark(clf, train_X, train_y, test_X, test_y)

        clf = DecisionTreeClassifier()
        benchmark(clf, train_X.toarray(), train_y, test_X.toarray(), test_y)
