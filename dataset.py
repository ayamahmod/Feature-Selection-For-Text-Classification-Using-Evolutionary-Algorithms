from urllib2 import HTTPError
from urllib2 import quote
from urllib2 import urlopen
import os, sys, tarfile
from tempfile import NamedTemporaryFile


datasets = {
    "reuters": "http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz",
    "20newsgroups": "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz",
    "mini_newsgroups": "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz"
}

def fetch(dataname):
    path = os.path.join(os.getcwd(), dataname)
    if os.path.isdir(path):
        print('data already downloaded')
        return path

    if dataname not in datasets:
        pass
    url = datasets[dataname]
    try:
        data_url = urlopen(url)
    except HTTPError as e:
        if e.code == 404:
            e.msg = "url {0} not found".format(url)
        raise

    try:
        print('downloading file...')
        f = NamedTemporaryFile(delete=False)
        f.write(data_url.read())
        f.close()
        print('extract files...')
        with tarfile.open(f.name) as tar:
            tar.extractall(path)
    except:
        raise
    finally:
        os.unlink(f.name)
    data_url.close()
    return path
