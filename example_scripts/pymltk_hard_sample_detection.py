#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard sample detection.
"""

#===============================================================================
# Script Name: pymltk_hard_sample_detection.py
#
# Author: David Piñeyro
# Contact info: dapineyro.dev@gmail.com
# Date: 2020-03-16
# Version: 0.0.1
#
# License: GPL-3.
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Usage instructions: see python3 pymltk_hard_sample_detection.py --help
#
# Description:
#   This script applies the hard sample detection function from pymltk to
#   quantify how many times each sample is correctly classified in many CV folds
#   of several models.
#===============================================================================

#===============================================================================
# IMPORTS
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
import sklearn.decomposition as decomp
import sklearn.impute as impute
import sklearn.manifold as manifold
import sklearn.cluster as cluster
import scipy.cluster.hierarchy as hierarchy
import sklearn.model_selection as model_sel
import sklearn.feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from boruta import BorutaPy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from pymltk.handlers import methyl
from pymltk.transformers import remove
from pymltk.utils import preprocess

#===============================================================================
# GLOBAL VARIABLES
SCRIPT_DESCRIPTION = """
pymltk hard sample detector
Version: 0.0.1
Author: David Piñeyro
Date: 2021-03-16
License: GPL-3
"""
#===============================================================================
# FUNCTIONS
def arguments():
    """This function uses argparse functionality to collect arguments."""
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                description= SCRIPT_DESCRIPTION)
    parser.add_argument('-b', '--betas',
                        metavar='<str: path/to/betas.csv>',
                        type=str,
                        required=True,
                        help="""A betas matrix with CpGs in rows and samples
                                in columns. Colnames and a first unnamed column
                                with CpGs names are expected.""")
    parser.add_argument('-d', '--sample_sheet',
                        metavar='<str: path/to/sample_sheet.csv>',
                        type=str,
                        required=True,
                        help="""The sample sheet containing the samples labels
                                and possibly other clinical information. It
                                should contain, at least, the following columns:
                                'Sample_Name', `class_label`. It may contain
                                a previous header to be discarded using
                                `--skip_ss` argument. Samples
                                as rows and column names are expected (do not
                                skip them with `--skip_ss`.""")
    parser.add_argument('-s', '--skip_ss',
                        metavar='<int>',
                        type=int,
                        default = 0,
                        required = False,
                        help="""The number of rows to skip from the
                                `sample_sheet` before start to read. It is
                                intented to remove the normal header found in
                                illumina sample_sheet (e.g. --skip_ss 7).
                                Do not include colnames are as part of this
                                header. Default = 0.""")
    parser.add_argument('-c', '--class_label',
                        metavar='<str>',
                        type=str,
                        required=True,
                        help="""The column name of the variable to use as a
                                classification label, present in the
                                sample_sheet.""")
    parser.add_argument('-o', '--out_dir',
                        metavar='<str: path/to/out_dir>',
                        type=str,
                        required=True,
                        help="""Output directory were the results will be
                                placed.""")
    parser.add_argument('-p', '--cores',
                        metavar='<int>',
                        type=int,
                        default = 1,
                        required=False,
                        help="""Number of processor cores to be used in
                                parallel computations. Default = 1 (no
                                parallel computations).""")
    parser.add_argument('-fs1', '--fs_size_1',
                        metavar='<int>',
                        type=int,
                        default = 5000,
                        required=False,
                        help="""Number of features to be selected in the
                                first feature selection round.
                                Default = 5000.""")
    parser.add_argument('-fs2', '--fs_size_2',
                        metavar='<int>',
                        type=int,
                        default = 50,
                        required=False,
                        help="""Number of features to be selected in the
                                second feature selection round.
                                Default = 50.""")
    parser.add_argument('-r', '--random_seed',
                        metavar='<int>',
                        type=int,
                        default = 1234,
                        required=False,
                        help="""Random seed to use in stochastical
                                computations. Default = 1234.""")
    parser.add_argument('-cv', '--cross_val',
                        metavar='<int>',
                        type=int,
                        default = 10,
                        required=False,
                        help="""Cross-validation folds to use. Default = 10.""")
    parser.add_argument('-rcv', '--cv_rep',
                        metavar='<int>',
                        type=int,
                        default = 1,
                        required=False,
                        help="""Cross-validations repeats to use in
                                repeated cross-validation. Default = 1
                                (no repeated cross-validation).""")

    args = parser.parse_args()
    print('Betas file:', args.betas)
    print('Sample sheet files:', args.sample_sheet)
    print(f'Skip {args.skip_ss} rows from sample sheet before reading.')
    print('Column to use as class labels:', args.class_label)
    print('Output directory:', args.out_dir)
    print('Cores to use in parallel computations:', args.cores)
    print(f'Number of features to select in a first round: {args.fs_size_1}')
    print(f'Number of features to select in a second round: {args.fs_size_2}')
    print(f'The random seed for stochastic calculations will be {args.random_seed}')
    print(f'The CV strategy will be {args.cv_rep} x {args.cross_val}-Fold CV')

    return args

#===============================================================================
# CLASSES

#===============================================================================
# MAIN
def main_program():
    """
    This is the main program.
    """
    # Collect command line arguments.
    args = arguments()

    ### Data loading.
    print('[MSG] Loading methylation data...')
    discovery = methyl.MethylData(args.betas,
                                  args.sample_sheet,
                                  args.class_label,
                                  skip_ss = args.skip_ss)
    print('[MSG] Methylation data loaded.')

    X = discovery.betas_.copy(deep = True)
    y = discovery.y_.copy(deep = True)

    ### Perparation of the Pipeline.
    ## Data Preprocessing.
    # Initial feature selection by NaN proporiton over max_nan_prop
    # (default 0.1).
    remove_nan = ('remove_nan',
                  remove.RemoveFeaturesByNaN(max_nan_prop = 0.05,
                                             verbose = False)
    )
    # Remove highly correlated features (default abs(0.9)).
    remove_cor = ('remove_cor',
                  remove.RemoveCorrelatedFeatures(random_seed = args.random_seed,
                                                  verbose = False)
    )
    # Imputation
    imputation = (
        'simple_imp',
        impute.SimpleImputer(missing_values = np.nan,
                            # add_indicator = True,
                             strategy = 'median'
                            )
    )
    # FS 1
    limma = (
        'Limma',
        remove.LimmaFS(k = args.fs_size_1, to_m_vals = True)
    )
    # FS 2
    elasticnet = (
        'ElasticNet_fs_2',
        fs.SelectFromModel(
            estimator = SGDClassifier(loss = "log",
                                      penalty = "elasticnet",
                                      max_iter = 10000,
                                      tol = 1e-3,
                                      n_jobs = 1,
                                      class_weight = 'balanced',
                                      random_state = args.random_seed),
            threshold = -np.inf,
            max_features = args.fs_size_2)
    )

    # Classificator
    rfc = (
        'RandomForest',
        RandomForestClassifier(n_estimators = 100,
                               random_state = args.random_seed,
                               class_weight = 'balanced',
                               max_features = 'auto',
                               n_jobs = 1,
                               max_depth = 5)
    )
    lrc = (
        'LR',
        LogisticRegression(solver = 'lbfgs',
                           random_state = args.random_seed)
    )
    xgboost = (
        'xgboost',
        xgb.XGBClassifier(n_estimators = 1000,
                          max_depth = 3,
                          learning_rate = 0.1,
                          gamma = 0.1,
                          subsample = 0.5,
                          reg_alpha = 0.1,
                          reg_lambda = 2,
                          scale_pos_weight = 0.5,
                          random_state = args.random_seed)
    )
    # Estimator
    estimator_rfc = Pipeline([remove_nan,
                             imputation,
                             limma,
                             remove_cor,
                             elasticnet,
                             rfc])
    estimator_lrc = Pipeline([remove_nan,
                             imputation,
                             limma,
                             remove_cor,
                             elasticnet,
                             lrc])
    estimator_xgb = Pipeline([remove_nan,
                             imputation,
                             limma,
                             remove_cor,
                             elasticnet,
                             xgboost])
    # CV implementations available.
    rskfold = model_sel.RepeatedStratifiedKFold(n_splits = args.cross_val,
                                                n_repeats = args.cv_rep,
                                                random_state = args.random_seed)
    ### Hard sample detection
    hsd_rfc = preprocess.hard_sample_detector(estimator_rfc, X, y, groups = y,
                                              cv = rskfold, n_jobs = args.cores)
    hsd_lrc = preprocess.hard_sample_detector(estimator_lrc, X, y, groups = y,
                                              cv = rskfold, n_jobs = args.cores)
    hsd_xgb = preprocess.hard_sample_detector(estimator_xgb, X, y, groups = y,
                                              cv = rskfold, n_jobs = args.cores)
    print('Results of RandomForestClassifier')
    print(hsd_rfc)
    print()
    print('Results of LogisticRegression')
    print(hsd_lrc)
    print()
    print('Results of XGBoost')
    print(hsd_xgb)

    with open(os.path.join(args.out_dir, 'hard_sample_detection_rfc.txt'), 'w') as f:
        f.write('Results of RandomForestClassifier\n')
        f.write('Samples used:\n')
        f.write(str(X.index.values) + '\n')
        f.write(y.to_string() + '\n\n')
        f.write(f'CV iterations: {args.cross_val * args.cv_rep}\n\n')
        for k, v in hsd_rfc.items():
            f.write(str(k) + '\n')
            f.write(str(v) + '\n\n')

    with open(os.path.join(args.out_dir, 'hard_sample_detection_lrc.txt'), 'w') as f:
        f.write('Results of LogisticRegression\n')
        f.write('Samples used:\n')
        f.write(str(X.index.values) + '\n')
        f.write(y.to_string() + '\n\n')
        f.write(f'CV iterations: {args.cross_val * args.cv_rep}\n\n')
        for k, v in hsd_lrc.items():
            f.write(str(k) + '\n')
            f.write(str(v) + '\n\n')

    with open(os.path.join(args.out_dir, 'hard_sample_detection_xgb.txt'), 'w') as f:
        f.write('Results of XGBoost\n')
        f.write('Samples used:\n')
        f.write(str(X.index.values) + '\n')
        f.write(y.to_string() + '\n\n')
        f.write(f'CV iterations: {args.cross_val * args.cv_rep}\n\n')
        for k, v in hsd_xgb.items():
            f.write(str(k) + '\n')
            f.write(str(v) + '\n\n')
#===============================================================================
# Conditional to run the script
if __name__ == '__main__':
    main_program()
