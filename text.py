import re
from collections import Counter
from time import time
from sklearn.utils.extmath import density
import argparse
from collections import defaultdict, Counter
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk import stem
from nltk.corpus import stopwords
from reader import ReutersReader, NewsgroupsReader

ten_most = ['earn', 'acq',
            'money-fx', 'grain',
            'crude', 'trade',
            'interest', 'ship',
            'wheat', 'corn']

k_features = [500, 1000, 2000, 5000, -1]


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


class TfidfBuilder(object):
    def __init__(self, min_f=3):
        self.tokenizer = StemTokenizer()
        self.min_f = min_f
        self.stops = stopwords.words('english')

    def bagOfWords(self, data):
        bags = []
        for doc in data:
            tokens = self.tokenizer(doc)
            counts = Counter(tokens)
            size = float(sum([v for k, v in counts.items()]))
            bag = {k: v/size for k, v in counts.items()}
            bags.append(bag)
        return bags


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
    selector = SelectKBest(chi2, k=k)
    selector.fit(train_X, train_y)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X


def build_word_vector(data):
    def get_word_vector(text):
        stop = stopwords.words('english')
        tokens = nltk.word_tokenize(text)
        tokens = [t.lower() for t in tokens if t.lower() not in stop]
        vector = nltk.FreqDist(tokens)
        vector = [(k, v) for k, v in vector.iteritems() if v >= 3]
        return vector
    return [get_word_vector(d[1]) for d in data]


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


def benchmark(clf, train_X, train_y, test_X, test_y, encoder):
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
    for label, s in zip(labels, s):
        print("{0}:\t\t{1}".format(label, s))

    # print("confusion matrix:")
    # print(metrics.confusion_matrix(test_y, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


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
    clf = svm.LinearSVC()
    benchmark(clf, train_X, train_y, test_X, test_y, encoder)

    # clf = KNeighborsClassifier()
    # benchmark(clf, train_X, train_y, test_X, test_y, encoder)

    # clf = NearestCentroid()
    # benchmark(clf, train_X, train_y, test_X, test_y, encoder)

    # clf = MultinomialNB()
    # benchmark(clf, train_X, train_y, test_X, test_y, encoder)

    # clf = DecisionTreeClassifier()
    # benchmark(clf, train_X.toarray(), train_y, test_X.toarray(), test_y, encoder)

    # model = svmlight.learn(zip(train_y, train_X))
    # p = svmlight.classify(model, zip(test_y, test_X))
    # print(metrics.f1_score(test_y, p))
    # clf = svm.SVC(kernel='poly')
    # benchmark(clf, train_X, train_y, test_X, test_y, encoder)
    # clf = svm.SVC(gamma=7)
    # benchmark(clf, train_X, train_y, test_X, test_y, encoder)
    # for k in k_features:
    #     print("#"*80)
    #     if k > 0:
    #         print("select {0} features".format(k))
    #         train_X_sub, test_X_sub = select_features(train_X, train_y,
    #                                                   test_X, k)
    #     else:
    #         print("use all features")
    #         train_X_sub, test_X_sub = train_X, test_X
    #     clf = svm.LinearSVC()
    #     benchmark(clf, train_X_sub, train_y, test_X_sub, test_y, encoder)

        # for degree in range(1, 6):
        #     clf = svm.SVC(kernel='poly', degree=degree)
        #     benchmark(clf, train_X, train_y, test_X, test_y)

        # for gamma in [0.6, 0.8, 1, 1.2]:
        #     clf = svm.SVC(kernel='rbf', gamma=gamma)
        #     benchmark(clf, train_X, train_y, test_X, test_y, encoder)

        # clf = KNeighborsClassifier()
        # benchmark(clf, train_X, train_y, test_X, test_y)

        # clf = NearestCentroid()
        # benchmark(clf, train_X, train_y, test_X, test_y)

        # clf = MultinomialNB()
        # benchmark(clf, train_X, train_y, test_X, test_y)

        # clf = DecisionTreeClassifier()
        # benchmark(clf, train_X.toarray(), train_y, test_X.toarray(), test_y)
