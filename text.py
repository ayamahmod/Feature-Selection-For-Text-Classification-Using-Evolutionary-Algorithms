import re
from collections import Counter
from time import time
from sklearn.utils.extmath import density
import argparse
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk import stem
from nltk.corpus import stopwords
from reader import ReutersReader, NewsgroupsReader


class StemTokenizer(object):
    def __init__(self):
        self.wnl = stem.WordNetLemmatizer()
        self.word = re.compile('[a-zA-Z]+')

    def __call__(self, doc):
        tokens = re.split('\W+', doc.lower())
        # tokens = [token for sentence in nltk.sent_tokenize(doc)
        #           for token in nltk.word_tokenize(sentence)]
        tokens = [self.wnl.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if self.word.search(t)]
        # sentences = nltk.sent_tokenize(doc)
        # tokens = [self.wnl.lemmatize(word) for sentence in sentences
        #           for word in nltk.word_tokenize(sentence)]
        return tokens


def build_tfidf(train_data, test_data):
    stops = stopwords.words('english')
    stops.append('reuter')
    counter = CountVectorizer(tokenizer=StemTokenizer(),
                              stop_words=stops, min_df=3,
                              dtype=np.double)
    counter.fit(train_data)
    train_tf = counter.transform(train_data)
    test_tf = counter.transform(test_data)

    # tokens = counter.inverse_transform(train_tf)
    # for token in tokens[0:100]:
    #     print(token)

    # train_tf = normalize(train_tf, norm='l1', axis=1)
    # test_tf = normalize(test_tf, norm='l1', axis=1)
    # print(train_tf)
    transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    train_tfidf = transformer.fit_transform(train_tf)
    test_tfidf = transformer.transform(test_tf)
    # print(train_tfidf)
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


def print_benchmark(clf, kfeatures, score, scores, train_time, test_time):
    print(clf)
    print('features:\t{0}'.format(kfeatures))
    print("train time: {0:0.4f}s".format(train_time))
    print("test time:  {0:0.4f}s".format(test_time))
    print("f1-score:   {0:0.4f}".format(score))
    for label, s in scores:
        print("{0:10s}:\t\t{1:2.2f}".format(label, s*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="the data directory")
    parser.add_argument('type', choices=['reuters', 'newsgroups'],
                        help="choose the data type")
    args = parser.parse_args()

    print('read data')
    if args.type == 'reuters':
        reader = ReutersReader()
    elif args.type == 'newsgroups':
        reader = NewsgroupsReader()
    data = reader.read(args.path)

    print('filter data')
    train_text, train_label, test_text, test_label = reader.filter(data)

    print('encode labels')
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_label)
    test_y = encoder.transform(test_label)

    print("build tfidf")
    # tfidf = TfidfBuilder()
    # train_X = tfidf.bagOfWords(train_text)
    # print(train_X)
    train_X, test_X = build_tfidf(train_text, test_text)

    # def transform(X):
    #     X = X.toarray()
    #     n = X.shape[0]
    #     data = []
    #     for i in range(n):
    #         sample = X[i, :]
    #         vector = [(i, v) for i, v in enumerate(sample) if v != 0]
    #         data.append(vector)
    #     return data
    # train_X = transform(train_X)
    # test_X = transform(test_X)

    print("training")
    benchmarks = defaultdict(list)
    ks = [1, 15, 30, 45, 60]
    degrees = [1, 2, 3, 4, 5]
    gammas = [0.6, 0.8, 1, 1.2]
    k_features = [500, 1000, 2000, 5000, 'all']

    clfs = {}
    bestscores = {}
    bestk = {}

    def updatescore(name, clf, k, result):
        if name not in clfs:
            clfs[name] = clf
            bestscores[name] = result
            bestk[name] = k
        elif result[1] > bestscores[name][1]:
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

        # for g in gammas:
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
            clf = DecisionTreeClassifier()
            result = benchmark(clf, train_X_sub.toarray(), train_y,
                               test_X_sub.toarray(), test_y, encoder)
            print(result[1])
            updatescore('CART', clf, k, result)

    for key in clfs:
        print('-'*80)
        print(key)
        result = bestscores[key]
        k = bestk[key]
        print_benchmark(result[0], k, result[1],
                        result[2], result[3], result[4])
    # x = range(len(k_features))
    # for k, v in benchmarks.iteritems():
    #     print(v)
    #     print(x)
    #     plt.plot(x, v, )
