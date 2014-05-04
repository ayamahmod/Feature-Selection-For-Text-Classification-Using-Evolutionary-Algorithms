#!/usr/bin/env python
import re
from time import time
import argparse
from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from nltk import stem
from nltk.corpus import stopwords

from reader import ReutersReader, NewsgroupsReader
from dataset import datasets, fetch


class StemTokenizer(object):
    """
    Tokenizer for CountVectorizer with stemming support
    """
    def __init__(self):
        self.wnl = stem.WordNetLemmatizer()
        self.word = re.compile('[a-zA-Z]+')

    def __call__(self, doc):
        tokens = re.split('\W+', doc.lower())
        tokens = [self.wnl.lemmatize(t) for t in tokens]
        # tokens = [t for t in tokens if self.word.search(t)]
        return tokens


def build_tfidf(train_data, test_data):
    stops = stopwords.words('english')
    counter = CountVectorizer(tokenizer=StemTokenizer(),
                              stop_words=stops, min_df=3,
                              dtype=np.double)
    counter.fit(train_data)
    train_tf = counter.transform(train_data)
    test_tf = counter.transform(test_data)

    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    train_tfidf = transformer.fit_transform(train_tf)
    test_tfidf = transformer.transform(test_tf)
    return train_tfidf, test_tfidf


def select_features(train_X, train_y, test_X, k):
    if k == 'all':
        return train_X, test_X

    selector = SelectKBest(chi2, k=k)
    selector.fit(train_X, train_y)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X


def benchmark(clf, train_X, train_y, test_X, test_y, encoder):
    """
    benchmark based on f1 score
    """
    t0 = time()
    clf.fit(train_X, train_y)
    train_time = time() - t0

    t0 = time()
    pred = clf.predict(test_X)
    test_time = time() - t0

    score = metrics.f1_score(test_y, pred, average='micro')
    scores = metrics.f1_score(test_y, pred, average=None)
    counter = Counter(train_y)
    counter = [(k, v) for k, v in counter.iteritems()]
    counter.sort(key=lambda a: a[1], reverse=True)
    if len(counter) > 20:
        tops = [v[0] for v in counter[0:20]]
    else:
        tops = [v[0] for v in counter]
    labels = encoder.inverse_transform(tops)
    s = [scores[v] for v in tops]
    labeled_scores = zip(labels, s)

    return clf, score, labeled_scores, train_time, test_time


def print_benchmark(kfeatures, clf, score, scores, train_time, test_time):
    print(clf)
    print('features:\t{0}'.format(kfeatures))
    print("train time: {0:0.4f}s".format(train_time))
    print("test time:  {0:0.4f}s".format(test_time))
    print("f1-score:   {0:0.4f}".format(score))
    n = 0

    #find longest label, to make print pretty
    for label, _ in scores:
        if len(label) > n:
            n = len(label)

    for label, s in scores:
        tmpl = "{0:" + str(n) + "s}:\t{1:2.2f}"
        print(tmpl.format(label, s*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=datasets.keys(),
                        help="choose the data type")
    args = parser.parse_args()

    print('fetch data')
    path = fetch(args.type)

    print('read data')
    if args.type == 'reuters':
        reader = ReutersReader()
    else:
        reader = NewsgroupsReader()
    data = reader.read(path)

    print('filter data')
    train_text, train_label, test_text, test_label = reader.filter(data)

    print('encode labels')
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_label)
    test_y = encoder.transform(test_label)

    print("build tfidf")
    train_X, test_X = build_tfidf(train_text, test_text)

    print("training")
    ks = [1, 15, 30, 45, 60]
    k_features = [500, 1000, 2000, 5000, 'all']

    clfs = {}
    bestscores = {}
    bestk = {}

    def updatescore(name, clf, k, result):
        """
        update best parameter and scores
        """
        if name not in clfs:
            clfs[name] = clf
            bestscores[name] = result
            bestk[name] = k
        elif result[1] >= bestscores[name][1]:
            clfs[name] = clf
            bestscores[name] = result
            bestk[name] = k

    for k in k_features:
        print("select {0} features".format(k))
        train_X_sub, test_X_sub = select_features(train_X, train_y,
                                                  test_X, k)

        print('linear SVM')
        clf = svm.LinearSVC()
        result = benchmark(clf, train_X_sub, train_y,
                           test_X_sub, test_y, encoder)
        print(result[1])
        updatescore('svm linear', clf, k, result)

        print('svm rbf')
        clf = svm.SVC(kernel='rbf', gamma=1)
        result = benchmark(clf, train_X_sub, train_y,
                           test_X_sub, test_y, encoder)
        print(result[1])
        updatescore('svm rbf', clf, k, result)

        print('K-NN')
        for i in ks:
            clf = KNeighborsClassifier(n_neighbors=i)
            result = benchmark(clf, train_X_sub, train_y,
                               test_X_sub, test_y, encoder)
            print(result[1])
            updatescore('knn', clf, k, result)

        print('Rocchio')
        clf = NearestCentroid()
        result = benchmark(clf, train_X_sub, train_y,
                           test_X_sub, test_y, encoder)
        print(result[1])
        updatescore('Rocchio', clf, k, result)

        print('Bayes')
        clf = MultinomialNB()
        result = benchmark(clf, train_X_sub, train_y,
                           test_X_sub, test_y, encoder)
        print(result[1])
        updatescore('bayes', clf, k, result)

        print('CART')
        if k != 'all':
            #when k='all', it takes too much time. so remove it
            clf = DecisionTreeClassifier()
            result = benchmark(clf, train_X_sub.toarray(), train_y,
                               test_X_sub.toarray(), test_y, encoder)
            print(result[1])
            updatescore('CART', clf, k, result)

    #print all best benchmars for every method
    for key in clfs:
        print('-'*80)
        print(key)
        result = bestscores[key]
        k = bestk[key]
        print_benchmark(k, result[0], result[1],
                        result[2], result[3], result[4])
