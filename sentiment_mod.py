#!/usr/bin/env python
# coding: utf-8

# In[28]:


import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        

save_docs = open("pickles/documents.pickle", "rb")
saved_documents = pickle.load(save_docs)
save_docs.close()


save_word_features = open("pickles/5kwordfeatures.pickle", "rb")
saved_word_features = pickle.load(save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in saved_word_features:
        features[w] = (w in words)

    return features

save_featuresets = open("pickles/feature_sets.pickle", "rb")
featuresets = pickle.load(save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
    
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]


# In[29]:


save_classifier = open("pickles/NBclassifier.pickle", "rb")
classifier = pickle.load(save_classifier)
save_classifier.close()

save_classifier = open("pickles/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(save_classifier)
save_classifier.close()


save_classifier = open("pickles/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(save_classifier)
save_classifier.close()


save_classifier = open("pickles/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(save_classifier)
save_classifier.close()



save_classifier = open("pickles/SGDClassifier_classifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(save_classifier)
save_classifier.close()


save_classifier = open("pickles/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(save_classifier)
save_classifier.close()



voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


# In[ ]:




