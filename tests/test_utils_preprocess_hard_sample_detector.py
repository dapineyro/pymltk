
# Test

import warnings
import numbers
import time
from traceback import format_exc
import sys
import numpy as np
import pandas as pd
import sklearn.model_selection as model_sel
from joblib import Parallel, logger, delayed

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _check_fit_params
from sklearn.model_selection._split import check_cv
from pymltk.utils.preprocess import hard_sample_detector
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rskfold = model_sel.RepeatedStratifiedKFold(n_splits = 5,
                                            n_repeats = 10,
                                            random_state = 1234)

X = np.array([[10, 5, 2], [11, 4,3], [9, 5, 1], [9, 6,2], [12, 4, 1], [10, 3, 1], [3, 1, 8],
              [2,2,7], [3,2,9], [1,2,6], [3,3,6], [1,2,5]])
y = np.array(['a' for i in range(6)] + ['b' for i in range(6)])

hs = hard_sample_detector(rf, X, y, groups = y, cv = rskfold, n_jobs = 2, return_train_pred = True, verbose = 10)

print(hs['train_success'])
print(hs['test_success'])

y[4] = 'b'

hs2 = hard_sample_detector(rf, X, y, groups = y, cv = rskfold, n_jobs = 2, return_train_pred = True, verbose = 10)
print(hs2['train_success'])
print(hs2['test_success'])
