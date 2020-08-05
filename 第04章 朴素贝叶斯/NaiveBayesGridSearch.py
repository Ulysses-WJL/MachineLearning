#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: ulysses
Date: 2020-08-05 10:52:09
LastEditTime: 2020-08-05 12:48:32
LastEditors: ulysses
Description: 
'''
from time import time
from pprint import pprint
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

print("Loading 20 newsgroups dataset for categories:")

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

print("train documents: {} ".format(len(data_train.filenames)))
print("test documents: {} ".format(len(data_test.filenames)))
print("{} categories".format(len(data_train.target_names)))
print()

# 构建 pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

paramerters = {
    'vect__max_df': (0.5, 0.75, 1.0),  # 滤去出现频率过高的词
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # 一元组, 二元组
    'vect__token_pattern': (r"\b\w\w+\b",  # 分词方式
                           r'\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b'),
    'vect__stop_words': (None, 'english'),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf_norm': ('l1', 'l2'),
    'clf__alpha': (1.0, 0.5),  # Laplace/Lidstone/no smoothing
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, paramerters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("paramerters:")
    pprint(paramerters)
    t0 = time()

    grid_search.fit(data_train.data, data_train.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Train Best score: {:.3f}".format(grid_search.best_score_))

    best_params = grid_search.best_params_
    for param_name in sorted(paramerters.keys()):
        print("\t{}: {}".format(param_name, best_params[param_name]))
    print("test socre", grid_search.score(
                        data_test.data, data_test.target))
