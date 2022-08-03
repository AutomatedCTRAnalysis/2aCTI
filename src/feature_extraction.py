from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from skmultilearn.adapt import MLkNN
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
     
from nltk.stem.snowball import SnowballStemmer

import pandas as pd
import json
import os
import re

import gensim 
from gensim.models import Word2Vec
import gensim.downloader

import pickle 

import nltk
from nltk.tokenize import word_tokenize

import spacy
import matplotlib
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
from ast import literal_eval
from tqdm import tqdm

import sklearn.metrics
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import flair

def feature_extraction(featureExtract, X_train_text, X_test_text, average = False, embedding_type = None, weighted=False):
    if featureExtract in ['CountVectorizer', 'TfIdfVectorizer']:
        if featureExtract == 'CountVectorizer':
            fe = CountVectorizer(analyzer ='word', stop_words ='english', lowercase = True, min_df = 2, max_df = 0.99) # if words used less than 0.001 % and in less than 2 documents --> ignore  
        else:
            fe = TfidfVectorizer(analyzer = 'word', stop_words='english', lowercase=True, min_df = 2, max_df=0.99)
        
        X_train = fe.fit_transform(X_train_text)
        X_train = pd.DataFrame(X_train.toarray(), columns = fe.get_feature_names()) 
        X_test = fe.transform(X_test_text)
        X_test = pd.DataFrame(X_test.toarray(), columns = fe.get_feature_names())
    
    else:
        if embedding_type is None:
            raise ValueError("Missing embedding method")
        model = embedding_type
        
        # sent is tokenised sentence on which we do the embedding      
        def get_embeddings(sent, tfidf, model, average=False):     
            # if text not in vocab:
            words_in_vocab = [word for word in sent if word in model]
            if not words_in_vocab:
                return np.zeros_like(model['the'])
            emb = model[words_in_vocab]
            if tfidf is not None:
                weights = np.ones((len(words_in_vocab), 1))
                for i, word in enumerate(words_in_vocab):
                    if word in fe.get_feature_names():
                        weights[i, 0] = tfidf.get(word, 1)
                emb *= weights
            return np.mean(emb, axis=0) if average else np.sum(emb, axis=0)
        
        #perform tokenisation
        if weighted:
            fe = TfidfVectorizer(analyzer = 'word', stop_words='english', lowercase=True, min_df = 2, max_df=0.99)
            X_train_tfidf = pd.DataFrame(fe.fit_transform(X_train_text).toarray(), columns = fe.get_feature_names()) 
            X_test_tfidf = pd.DataFrame(fe.fit_transform(X_test_text).toarray(), columns = fe.get_feature_names()) 
        else:
            X_train_tfidf = None
            X_test_tfidf = None
        X_train = []
        
        for i, text in tqdm(enumerate(X_train_text)):
            tokens = nltk.word_tokenize(text)
            tfidf = X_train_tfidf.loc[i] if weighted else None
            embeddings = get_embeddings(tokens, tfidf, model)
            X_train.append(embeddings.tolist())
        X_test = []
        
        for i, text in tqdm(enumerate(X_test_text)):
            tokens = nltk.word_tokenize(text)
            tfidf = X_test_tfidf.loc[i] if weighted else None
            embeddings = get_embeddings(tokens, tfidf, model)
            X_test.append(embeddings.tolist())
    return X_train, X_test

def evaluation(Y_pred, Y_test):
    macro_precision = precision_score(Y_test, Y_pred, average ='macro')
    micro_precision = precision_score(Y_test, Y_pred, average ='micro')
    macro_recall = recall_score(Y_test, Y_pred, average='macro')
    micro_recall = recall_score(Y_test, Y_pred, average='micro')
    macro_fscore = fbeta_score(Y_test, Y_pred, beta=0.5, average ='macro')
    micro_fscore = fbeta_score(Y_test, Y_pred, beta=0.5, average ='micro')
    l_metric = ['macro precision', 'micro precision', 'macro recall', 'micro recall', 'macro fscore', 'micro fscore']
    l_result = [macro_precision, micro_precision, macro_recall, micro_recall, macro_fscore, micro_fscore]
    df_res = pd.DataFrame({'metric': l_metric, 'result': l_result})
    return df_res

        
        