#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GridSearch and final model training.
"""

#===============================================================================
# Script Name: pymltk_gridsearch.py
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
# Usage instructions: see python3 pymltk_gridsearch.py --help
#
# Description:
#   This script will optimize each of the selected best models using grid
#   search creating and saving an object with the best model to test with
#   the test dataset.
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

import pickle

#===============================================================================
# GLOBAL VARIABLES
SCRIPT_DESCRIPTION = """
pymltk gridsearch
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
    parser.add_argument('-t', '--test_size',
                        metavar='<float>',
                        type=float,
                        default = 0.3,
                        required=False,
                        help="""Test proportion for a training / test
                                splitting. Default = 0.3.""")
    parser.add_argument('-stl', '--strat_lab',
                        metavar='<str>',
                        type=str,
                        default = '',
                        required=False,
                        help="""An `class_label` stratified training / test is
                                performed by default. To select other columns to
                                stratify for, use this parameter. This parameter
                                accepts only a string, composed of the column names
                                to be used in the stratification, separated by ','
                                character. No spaces are allowed. The first column
                                name is taken as the most important and the
                                last as the least important. Please, use the
                                `class_label` as the first element of the list.
                                Default = '' (only the class_label will be used
                                for stratification). Example:
                                -stl Response,Age_group,Gender. Note: Neither
                                spaces nor other special characters are allowed,
                                so check your colnames of the sample_sheet.""")
    parser.add_argument('-sc', '--scoring',
                        metavar='<str>',
                        type=str,
                        default = 'balanced_accuracy',
                        required=False,
                        help="""A string indicating the scoring strategy for CV.
                                See 'https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter'
                                for valid values.
                                Default = 'balanced_accuracy'.""")

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
    print(f'{(1 - args.test_size) * 100} % training / ' +
          f'{args.test_size * 100} % test split will be performed.')
    if len(args.strat_lab) == 0:
        print('Splitting will be stratified using the class label ' +
              f'({args.class_label})')
    else:
        print('Splitting will be stratified using the following columns: ' +
              f"{str(args.strat_lab.split(','))}")
    print('Scoring strategy:', args.scoring)

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

    ### Prepare sample sheet for custom stratification process.
    if len(args.strat_lab) > 0:
        ss_strat= discovery.sample_sheet_.copy(deep = True)
        strat_lab = args.strat_lab.split(',')
        ss_strat = ss_strat.loc[:, strat_lab]

    else:
        ss_strat = discovery.y_.copy(deep = True)


    ### Prepare data for training & validation
    X = discovery.betas_.copy(deep = True)
    y = discovery.y_.copy(deep = True)
    print('Spliting training - test dataset')
    my_splits = preprocess.train_test_split_david(X,
                                                  y,
                                                  ss_strat,
                                                  test_size = args.test_size,
                                                  n_splits = 1,
                                                  random_state = args.random_seed,
                                                  max_diff_other = 0.1,
                                                  verbose = 0)
    X_train = my_splits[0][0]
    X_test = my_splits[0][1]
    y_train = my_splits[0][2]
    y_test = my_splits[0][3]
    cols_used_to_stratify = my_splits[0][4]

    # Description.
    print("X train:")
    print(X_train.head())
    print(f"X train shape: {X_train.shape}")
    print("y_train:")
    print(y_train.head())
    print(y_train.value_counts())

    print("X test:")
    print(X_test.head())
    print(f"X test shape: {X_test.shape}")
    print("y_test:")
    print(y_test.head())
    print(y_test.value_counts())

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
    # Data imputation.
    imputations = []
    imputations.append((
        'simple_imp',
        impute.SimpleImputer(missing_values = np.nan,
                            # add_indicator = True,
                             strategy = 'median'
                            )
    ))
    #imputations.append((
    #    'iterative_imp',
    #    impute.IterativeImputer(random_state = args.random_seed)
    #))
    #imputations.append((
    #    'knn_imp',
    #    impute.KNNImputer(missing_values = np.nan,
    #                      n_neighbors = 5,
    #                      weights = 'uniform'
    #                      )
    #))

    ### Feature Selection.
    fs_1 = []
    fs_1.append((
        'Limma',
        remove.LimmaFS(k = args.fs_size_1, to_m_vals = True)
    ))
    fs_1.append((
        'ANOVA',
        fs.SelectKBest(score_func = fs.f_classif, k = args.fs_size_1)
    ))
    fs_1.append((
        'chi2',
        fs.SelectKBest(score_func = fs.chi2, k = args.fs_size_1)
    ))
    #fs_1.append((
    #    'mutual_inf',
    #    fs.SelectKBest(score_func = fs.mutual_info_classif,
    #                   k = args.fs_size_1)
    #))
    fs_1.append((
        'ElasticNet_fs_1',
        fs.SelectFromModel(
            estimator = SGDClassifier(loss = "log",
                                      penalty = "elasticnet",
                                      max_iter = 10000,
                                      tol = 1e-3,
                                      n_jobs = 1,
                                      class_weight = 'balanced',
                                      random_state = args.random_seed),
            threshold = -np.inf,
            max_features = args.fs_size_1)
    ))

    fs_2 = []
    fs_2.append((
        'ANOVA_2',
        fs.SelectKBest(score_func = fs.f_classif, k = args.fs_size_2)
    ))
    fs_2.append((
        'chi2_2',
        fs.SelectKBest(score_func = fs.chi2, k = args.fs_size_2)
    ))
    fs_2.append((
        'mutual_inf_2',
        fs.SelectKBest(score_func = fs.mutual_info_classif,
                       k = args.fs_size_2)
    ))
    fs_2.append((
        'extra_trees',
        fs.SelectFromModel(
            estimator = ExtraTreesClassifier(n_estimators = 100,
                                             random_state = args.random_seed,
                                             max_depth = 5,
                                             max_features = 'auto',
                                             n_jobs = 1,
                                             class_weight = 'balanced_subsample',
                                             bootstrap = True,
                                             ccp_alpha = 0.005,
                                             oob_score = True,
                                             max_samples = 0.7),
            threshold = -np.inf,
            max_features = args.fs_size_2)
    ))
    fs_2.append((
        'L1_norm',
        fs.SelectFromModel(
            estimator = LinearSVC(C = 0.1,
                                  penalty = 'l1',
                                  dual = False,
                                  class_weight = 'balanced',
                                  max_iter = 10000,
                                  random_state = args.random_seed),
            threshold = -np.inf,
            max_features = args.fs_size_2)
    ))
    fs_2.append((
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
    ))
    fs_2.append((
        'RFE',
        fs.RFE(estimator = SVC(kernel = 'linear',
                               class_weight = 'balanced',
                               gamma = 'scale',
                               random_state = args.random_seed),
               n_features_to_select = args.fs_size_2
              )
    ))
    fs_2.append((
        'boruta',
        BorutaPy(estimator = RandomForestClassifier(n_estimators = 100,
                                                    random_state = args.random_seed,
                                                    class_weight = 'balanced',
                                                    max_depth = 5,
                                                    max_features = 'auto',
                                                    n_jobs = 1),
                 n_estimators = 'auto',
                 random_state = args.random_seed))
    )
    fs_2.append((
        'VotingFS',
        remove.VotingFS(selectors = [fs_2[i] for i in [0, 1, 3, 5]],
                        k = args.fs_size_2,
                        v = 2)
    ))

    ### Classification.
    models = []
    models.append((
        'LR',
        LogisticRegression(solver = 'lbfgs',
                           random_state = args.random_seed)
    ))
    models.append((
        'LDA',
        LinearDiscriminantAnalysis()
    ))
    models.append((
        'KNN',
        KNeighborsClassifier()
    ))
    models.append((
        'CART',
        DecisionTreeClassifier(random_state = args.random_seed)
    ))
    models.append((
        'NB',
        GaussianNB()
    ))
    models.append((
        'SVM',
        SVC(C = 2,
            class_weight = 'balanced',
            kernel = 'rbf',
            gamma = 'scale',
            random_state = args.random_seed)
    ))
    models.append((
        'ElasticNet',
        SGDClassifier(loss = "log",
                      penalty = "elasticnet",
                      max_iter = 1000,
                      tol = 1e-3,
                      n_jobs = 1,
                      class_weight = 'balanced',
                      random_state = args.random_seed)
    ))
    models.append((
        'MLP',
        MLPClassifier(hidden_layer_sizes = (args.fs_size_2 * 2,
                                            args.fs_size_2,
                                            int(args.fs_size_2 / 4)),
                      activation = 'relu',
                      solver = 'lbfgs',
                      alpha = 0.01,
                      max_iter = 1000,
                      random_state = args.random_seed)
    ))
    models.append((
        'AdaBoost',
        AdaBoostClassifier(
            base_estimator = DecisionTreeClassifier(
                max_depth = 5,
                random_state = args.random_seed,
                class_weight = 'balanced'),
        n_estimators = 100,
        random_state = args.random_seed)
    ))
    models.append((
        'RandomForest',
        RandomForestClassifier(n_estimators = 100,
                               random_state = args.random_seed,
                               class_weight = 'balanced',
                               max_features = 'auto',
                               n_jobs = 1,
                               max_depth = 5)
    ))
    models.append((
        'GradientBoosting',
        GradientBoostingClassifier(n_estimators = 100,
                                   random_state = args.random_seed,
                                   ccp_alpha = 0.005,
                                   max_features = 'auto')
    ))
    models.append((
        'xgboost',
        xgb.XGBClassifier(n_estimators = 100,
                         # max_depth = 3,
                         # learning_rate = 0.1,
                         # gamma = 0.1,
                         # subsample = 0.5,
                         # reg_alpha = 0.1,
                         # reg_lambda = 2,
                         # scale_pos_weight = 0.5,
                          random_state = args.random_seed)
    ))

    # Some scoring functions to use as alternative of accuracy.
    f1_scorer = make_scorer(f1_score)
    kappa_scorer = make_scorer(cohen_kappa_score)
    mcc_scorer = make_scorer(matthews_corrcoef)

    # CV implementations available.
    rskfold = model_sel.RepeatedStratifiedKFold(n_splits = args.cross_val,
                                                n_repeats = args.cv_rep,
                                                random_state = args.random_seed)


    #### Grid search optimization.
    ## Model 1: simple_impute + Limma + VotingFS
    choice_01 = [remove_nan,
                 imputations[0],
                 fs_1[0],
                 remove_cor,
                 fs_2[8],
                 models[10]]
    name_fs_1 = choice_01[2][0]
    name_fs_2 = choice_01[4][0]
    name_cls = choice_01[5][0]
    model_01 = Pipeline(choice_01)
    param_grid_01 = {
        'Limma__k': [int(args.fs_size_1/10), int(args.fs_size_1/2), args.fs_size_1],
        'VotingFS__v': [1, 2],
        'GradientBoosting__min_samples_split': [2, 4, 6],
        'GradientBoosting__max_depth': [2, 3, 5, 7],
        'GradientBoosting__learning_rate': [0.01, 0.1, 0.5],
        'GradientBoosting__n_estimators': [50, 100, 500],
        'GradientBoosting__tol': [1e-5, 1e-4, 1e-3],
        'GradientBoosting__ccp_alpha': [0.0, 0.001, 0.01]
    }
    search_grid_01 = model_sel.GridSearchCV(estimator = model_01,
                                           param_grid = param_grid_01,
                                           scoring = args.scoring,
                                           n_jobs = args.cores,
                                           cv = rskfold,
                                           error_score = 'raise'
                                           )
    search_grid_01.fit(X_train, y_train)
    choice_01_gridname = (name_fs_1 + '_' + name_fs_2 + '_' +
                          name_cls + '_gridsearch.pickle')
    with open(os.path.join(args.out_dir, choice_01_gridname), 'wb') as f:
        pickle.dump(search_grid_01, file = f)

    print('*' * 80)
    print(f'Grid search optimization for {choice_01_gridname}:')
    print(f'Best parameters (CV score={search_grid_01.best_score_}):')
    print(search_grid_01.best_params_)
    # Predicting using the best estimator.
    grid_01_pred = search_grid_01.predict(X_test)
    print('Predictions using the best estimator:')
    print(grid_01_pred)
    print('Ground truth:')
    print(y_test.to_string())
    print('Accuracy:',
          accuracy_score(y_test,
                         grid_01_pred))
    print('Balanced accuracy:',
          balanced_accuracy_score(y_test,
                                  grid_01_pred))
    print('Confusion matrix:\n',
          confusion_matrix(y_test,
                           grid_01_pred))
    print('Classification report:\n',
          classification_report(y_test,
                                grid_01_pred))
    print('Kappa statistic:',
          cohen_kappa_score(y_test,
                            grid_01_pred))

    search_grid_01_cv_r = pd.DataFrame(search_grid_01.cv_results_)
    search_grid_01_cv_r.to_csv(os.path.join(args.out_dir,
                                            choice_01_gridname + '.csv')
                              )
    choice_01_bestname = (name_fs_1 + '_' + name_fs_2 + '_' +
                          name_cls + '_bestmodel.pickle')
    with open(os.path.join(args.out_dir, choice_01_bestname), 'wb') as f:
        pickle.dump(search_grid_01.best_estimator_, file = f)
    print('Best estimator description:')
    print(search_grid_01.best_estimator_)

    print('Selected CpGs:')
    remove_nan_sel = search_grid_01.best_estimator_.named_steps['remove_nan'].features_to_keep_
    remove_nan_preselected = X_train.columns[remove_nan_sel]
    # FS_1:
    fs_1_sel = search_grid_01.best_estimator_.named_steps[name_fs_1].get_support()
    b_vals_preselected = remove_nan_preselected[fs_1_sel]
    # remove_cor:
    cor_sel = search_grid_01.best_estimator_.named_steps['remove_cor'].features_to_keep_
    b_vals_preselected_2 = b_vals_preselected[cor_sel]
    if name_fs_2 == 'boruta':
        f_selection = search_grid_01.best_estimator_.named_steps[name_fs_2].support_
        print(b_vals_preselected_2[f_selection])
    else:
        f_selection = search_grid_01.best_estimator_.named_steps[name_fs_2].get_support()
    selected_features = b_vals_preselected_2[f_selection]
    print(selected_features.format())
    with open(os.path.join(args.out_dir,
                           choice_01_bestname + '.selected_features.txt'), 'w') as f:
        f.write(str(selected_features.format()))
    print('*' * 80)

#===============================================================================
# Conditional to run the script
if __name__ == '__main__':
    main_program()
