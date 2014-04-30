import os
from HTMLParser import HTMLParser
import fnmatch
import re
from itertools import groupby
from sklearn.cross_validation import train_test_split

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


class NewsgroupsParser():
    META = {
        'newsgroups': 'newsgroup',
        'Path': 'path',
        'From': 'from',
        'Subject': 'sub',
        'Message-ID': 'id',
        'Sender': 'sender',
        'Organization': 'org',
        'References': 'ref',
        'Distribution': 'dist',
        'Date': 'date',
    }
    LINES = 'Lines'

    def __init__(self):
        pass

    def parse(self, text):
        doc = {}
        lines = text.split('\n')
        index = 0
        for i in range(len(lines)):
            line = lines[i]
            token = line.split(':')
            if len(token) == 1:
                pass
            else:
                a = token[0]
                if a == self.LINES:
                    index = i+1
                    break
                elif a in self.META:
                    doc[self.META[a]] = token[1]

        body = " ".join(lines[index:])
        doc['body'] = body
        return doc


class ReutersReader():
    def read(self, datapath):
        data = []
        for root, _dirnames, filenames in os.walk(datapath):
            for filename in fnmatch.filter(filenames, '*.sgm'):
                path = os.path.join(root, filename)
                parser = ReutersParser()
                with open(path, 'r') as f:
                    text = f.read()
                    d = parser.parse(text)
                    data += d
        return data

    def filter(self, docs):
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
        print('train texts:{0}, test texts: {1}'.format(len(train_text),
                                                        len(test_text)))
        return train_text, train_label, test_text, test_label


class NewsgroupsReader():
    def read(self, datapath):
        data = []
        for root, _dirnames, filenames in os.walk(datapath):
            for filename in filenames:
                path = os.path.join(root, filename)
                parser = NewsgroupsParser()
                with open(path, 'r') as f:
                    text = f.read()
                    text = unicode(text, errors='ignore')
                    d = parser.parse(text)
                    d['newsgroup'] = os.path.basename(root)
                    data.append(d)
        return data

    def filter(self, docs):
        train_text = []
        train_label = []
        test_text = []
        test_label = []
        for g, v in groupby(docs, key=lambda d: d['newsgroup']):
            traindocs, testdocs = train_test_split(list(v))
            print("{0} split to train: {1}, test: {2}".format(g,
                                                              len(traindocs),
                                                              len(testdocs)))
            for doc in traindocs:
                # texts = [v for k, v in doc.iteritems() if k != 'newsgroup']
                # text = " ".join(texts)
                text = doc['body']
                train_text.append(text)
                train_label.append(doc['newsgroup'])
            for doc in testdocs:
                # texts = [v for k, v in doc.iteritems() if k != 'newsgroup']
                # text = " ".join(texts)
                text = doc['body']
                test_text.append(text)
                test_label.append(doc['newsgroup'])
        return train_text, train_label, test_text, test_label
