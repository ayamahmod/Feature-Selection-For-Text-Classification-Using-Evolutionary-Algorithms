import os
from HTMLParser import HTMLParser
import re
import fnmatch
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

ten_most = ['earn', 'acq',
            'money-fx', 'grain',
            'crude', 'trade',
            'interest', 'ship',
            'wheat', 'corn']

k_features = [500, 1000, 2000, 5000, -1]


class ReutersParser(HTMLParser):
    """
    parse reuters 21578 dataset
    """
    REUTERS = "reuters"
    LEWISSPLIT = "lewissplit"
    TOPICS = "topics"
    TITLE = "title"
    BODY = "body"
    D = 'd'
    PLACES = 'places'
    PEOPLE = 'peopl'
    ORGS = 'orgs'
    EXCHANGES = 'exchanges'
    COMPANIES = 'companies'
    DATELINE = 'dateline'
    NEWID = 'newid'

    def __init__(self):
        HTMLParser.__init__(self)
        self._reset()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_d = 0
        self.in_topics = 0
        self.in_places = 0
        self.in_people = 0
        self.in_orgs = 0
        self.in_exchanges = 0
        self.in_companies = 0
        self.in_dateline = 0
        self.id = 0
        self.title = ""
        self.split = ""
        self.body = ""
        self.has_topics = ""
        self.dateline = ""
        self.topics = []
        self.places = []
        self.people = []
        self.orgs = []
        self.exchanges = []
        self.compnies = []

    def parse(self, text):
        self.docs = []
        self.feed(text)
        return self.docs

    def handle_starttag(self, tag, attrs):
        if tag == self.REUTERS:
            for attr in attrs:
                if attr[0] == self.LEWISSPLIT:
                    self.split = attr[1]
                elif attr[0] == self.TOPICS:
                    self.has_topics = attr[1]
                elif attr[0] == self.NEWID:
                    self.id = int(attr[1])
        elif tag == self.TOPICS:
            self.in_topics = 1
        elif tag == self.PLACES:
            self.in_places = 1
        elif tag == self.PEOPLE:
            self.in_people = 1
        elif tag == self.ORGS:
            self.in_orgs = 1
        elif tag == self.EXCHANGES:
            self.in_exchanges = 1
        elif tag == self.COMPANIES:
            self.in_companies = 1
        elif tag == self.DATELINE:
            self.in_dateline = 1
        elif tag == self.TITLE:
            self.in_title = 1
        elif tag == self.BODY:
            self.in_body = 1
        elif tag == self.D:
            self.in_d = 1

    def handle_endtag(self, tag):
        if tag == self.REUTERS:
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append({'split': self.split,
                              'has_topics': self.has_topics,
                              'id': self.id,
                              'topics': self.topics,
                              'places': self.places,
                              'people': self.people,
                              'orgs': self.orgs,
                              'exchanges': self.exchanges,
                              'companies': self.compnies,
                              'dateline': self.dateline,
                              'title': self.title,
                              'body': self.body})
            self._reset()
        elif tag == self.TOPICS:
            self.in_topics = 0
        elif tag == self.PLACES:
            self.in_places = 0
        elif tag == self.PEOPLE:
            self.in_people = 0
        elif tag == self.ORGS:
            self.in_orgs = 0
        elif tag == self.EXCHANGES:
            self.in_exchanges = 0
        elif tag == self.COMPANIES:
            self.in_companies = 0
        elif tag == self.DATELINE:
            self.in_dateline = 0
        elif tag == self.TITLE:
            self.in_title = 0
        elif tag == self.BODY:
            self.in_body = 0
        elif tag == self.D:
            self.in_d = 0

    def handle_data(self, data):
        if self.in_title:
            self.title = data
        elif self.in_body:
            self.body = data
        elif self.in_dateline:
            self.dateline = data
        elif self.in_d:
            if self.in_topics:
                self.topics.append(data)
            elif self.in_places:
                self.places.append(data)
            elif self.in_people:
                self.people.append(data)
            elif self.in_orgs:
                self.orgs.append(data)
            elif self.in_exchanges:
                self.exchanges.append(data)
            elif self.in_companies:
                self.companies.append(data)


class ReutersReader():
    def __init__(self, data_path):
        self.data_path = data_path

    def read(self):
        """Iterate doc by doc, yield a dict."""
        data = []
        for root, _dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, '*.sgm'):
                path = os.path.join(root, filename)
                parser = ReutersParser()
                with open(path, 'r') as f:
                    text = f.read()
                    d = parser.parse(text)
                    data += d
        return data


def filter_data(docs):
    traindocs = [doc for doc in docs
                 if doc['split'] == 'TRAIN' and doc['has_topics'] == 'YES'
                 and len(doc['topics']) == 1]
    testdocs = [doc for doc in docs
                if doc['split'] == 'TEST' and doc['has_topics'] == 'YES'
                and len(doc['topics']) == 1]

    traintopics = set()
    testtopics = set()
    for doc in traindocs:
        traintopics.update(doc['topics'])
    for doc in testdocs:
        testtopics.update(doc['topics'])
    topics = set.intersection(traintopics, testtopics)

    def extract_text(doc):
        items = doc['places'] + doc['people']
        items += doc['orgs'] + doc['exchanges'] + doc['companies']
        items.append(doc['title'])
        items.append(doc['dateline'])
        items.append(doc['body'])
        text = " ".join(items)
        return text

    def extract_data(docs):
        texts = []
        labels = []
        for doc in docs:
            n_topic = len(doc['topics'])
            if n_topic == 0:
                continue

            for i in range(n_topic):
                topic = doc['topics'][i]
                if topic in topics:
                    text = extract_text(doc)
                    texts.append(text)
                    labels.append(topic)
        return texts, labels

    train_text, train_label = extract_data(traindocs)
    test_text, test_label = extract_data(testdocs)

    print('all topics: {0}'.format(len(topics)))
    print('train docs:{0}, test docs {1}'.format(len(traindocs), len(testdocs)))
    print(len(train_text), len(test_text))
    return train_text, train_label, test_text, test_label


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
    tops = [v[0] for v in counter[0:10]]
    labels = encoder.inverse_transform(tops)
    s = [scores[v] for v in tops]
    print(zip(labels, s))

    # print("confusion matrix:")
    # print(metrics.confusion_matrix(test_y, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="the data directory")

    args = parser.parse_args()

    print('read data')
    reader = ReutersReader(args.path)
    data = reader.read()

    print('filter data')
    train_text, train_label, test_text, test_label = filter_data(data)

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
