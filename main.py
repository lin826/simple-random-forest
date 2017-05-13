import argparse
import sys
import logging
import csv
import numpy as np

from model_np import *
from plot import *
from preprocess import Preprocessor
from sklearn.decomposition import PCA

def load_csv(path):
    csv_file = open(path, 'rb')
    csv_reader = csv.reader(csv_file, delimiter=',')
    datas = np.array([data for data in csv_reader]).astype(np.float32)
    return datas

def get_model(args):
    logging.info('Model = %s' % args.model)
    if args.model == 'rf':
        return RandomForest()

def get_param(args):
    if args.model == 'rf':
        param_rf = [ float(p) for p in args.param_rf.split(',') ]
        return param_rf

def get_param_validate(args):
    if args.model == 'rf':
        return args.param_rf.split(':')

def preprocess(args, X, T):
    pre = Preprocessor()
    X_normal = X
    if args.pre == 'pca':
        logging.info('Preprocess with PCA(d = %d)' % args.deg)
        X_phi = pre.pca(X_normal, args.deg)
    elif args.pre == 'lda':
        logging.info('Preprocess with LDA(d = %d)' % args.deg)
        X_phi = pre.lda(X_normal, T, args.deg)
    return X_phi, pre

def main(args):
    model = get_model(args)
    if args.task == 'validate':
        X_Train = load_csv(args.train_X)
        T_Train = load_csv(args.train_T).flatten()
        X_Train_phi, phi = preprocess(args, X_Train, T_Train)

        logging.info('Training')
        model.validate(X_Train_phi, T_Train, params=get_param_validate(args))
    elif args.task == 'train':
        X_Train = load_csv(args.train_X)
        T_Train = load_csv(args.train_T).flatten()
        X_Train_phi, phi = preprocess(args, X_Train, T_Train)
        inds = range(len(X_Train))
        np.random.shuffle(inds)
        X_Train_phi = X_Train_phi[inds]
        T_Train = T_Train[inds]

        logging.info('Training')
        model.train(X_Train_phi, T_Train, param=get_param(args))

        train_acc = model.eval(X_Train_phi, T_Train)
        logging.info('Training Accuracy = %f' % train_acc)

        if args.test_X != None and args.test_T != None:
            X_Test = load_csv(args.test_X)
            T_Test = load_csv(args.test_T).flatten()
            X_Test_phi = phi.transform(X_Test)
            test_acc = model.eval(X_Test_phi, T_Test)
            logging.info ('Testing Accuracy = %f' % test_acc)

            print (test_acc)

        if args.save != None:
            model.save('%s' % args.save)
            logging.info('Model saved at %s' % args.save)

            phi.save('%s' % args.save + '_phi')
            logging.info('Model preprocessor saved at %s' % args.save + '_phi')

    elif args.task == 'plot':
        model.load(args.load)
        logging.info('Model loaded from %s' % args.load)
        logging.info('Plotting')
        l_tree = model.model.estimators_
        plot_decision_tree(l_tree,args.dot)

    elif args.task == 'dt_eval':
        phi = Preprocessor()
        phi.load(args.load + '_phi')

        X_Test = load_csv(args.test_X)
        T_Test = load_csv(args.test_T).flatten()
        X_Test_phi = phi.transform(X_Test)

        model.load(args.load)
        logging.info('Model loaded from %s' % args.load)
        logging.info('Decision Tree Evaluating')
        l_tree = model.model.estimators_
        for tree in l_tree:
            test_acc = tree.score(X_Test_phi, T_Test)
            print('%f' % test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_X', help='training data X', type=str, default='data/X_train.csv')
    parser.add_argument('--train_T', help='training data T', type=str, default='data/T_train.csv')
    parser.add_argument('--test_X', help='testing data X', type=str, default='data/X_test.csv')
    parser.add_argument('--test_T', help='testing data T', type=str, default='data/T_test.csv')

    parser.add_argument('--load', help='model load from', type=str)
    parser.add_argument('--save', help='model save to', type=str)

    parser.add_argument('--pre', help='preprocess type', type=str, choices=['pca', 'lda'], default='pca')
    parser.add_argument('--deg', help='degree for preprocess', type=int, default=10)
    parser.add_argument('--task', help='task type', type=str, choices=['validate', 'train', 'plot', 'dt_eval'], default='validate')
    parser.add_argument('--model', help='model type', type=str, choices=['rf'], default='rf')

    parser.add_argument('--param_rf', help='parameter for rf', type=str, default='100,1000,0.5')
    parser.add_argument('--verbose', help='log on/off', type=bool, default=False)

    parser.add_argument('--dot', help='plotting dot save to', type=str, default='../data/treepic.dot')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    else:
        logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.ERROR)

    main(args)
