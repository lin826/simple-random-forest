import logging
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm

from sklearn.ensemble import RandomForestClassifier

import cPickle as pickle

class Model(object):
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"), True)

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

    def test(self, X):
        return self.model.predict(X)

    def eval(self, X, T):
        y = self.test(X).astype(np.int32)
        t = T.astype(np.int32)
        return float(np.equal(y, t).sum()) / len(t)

class RandomForest(Model):
    def train(self, X, T, param):
        logging.info('X = [%d, %d], T = [1]' % (X.shape[0], X.shape[1]))
        n_train = int(float(len(X) * param[2]))
        for i in range(1, 6, 1):
            t_ = T[:n_train]
            #print (t_ == i)
            print ('%d = %d' % (i, (t_ == i).sum()))
        self.model = RandomForestClassifier(n_estimators=int(param[0]), min_samples_leaf=int(param[1]))
        self.model.fit(X[:n_train], T[:n_train])

    def validate(self, X, T, params):
        logging.info('X = [%d, %d], T = [1]' % (X.shape[0], X.shape[1]))

        stat = []

        for param_ in params:
            param = [ float(p) for p in param_.split(',') ]

            n_train = int(float(len(X) * param[2]))
            logging.info('[%d, %d, %f], n_train = %d' % (param[0], param[1], param[2], n_train))

            model = RandomForestClassifier(n_estimators=int(param[0]), min_samples_leaf=int(param[1]))
            scores = (cross_val_score(model, X[:n_train], T[:n_train], cv=3, n_jobs=1))
            score = np.mean(scores)

            logging.info('Validation accuracy = %f' % score)
            stat.append((param[0], param[1], param[2], score))
