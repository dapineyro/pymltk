"""
Collection of useful utilities to preprocess data.
"""
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

def betas2m(betas):
    """
    This function transforms a 2D numpy.ndarray object of beta-values into
    a 2D numpy.ndarray of m-values. This function also works with
    pandas.core.frame.DataFrame objects.
    Arguments:
        betas [numpy.ndarray/pandas.core.frame.DataFrame]: beta-values to
            be transformed into m-values. It can contain NaN (missing values),
            but all the values should be of dtype = float64.
    Output:
        m_values [numpy.ndarray/pandas.core.frame.DataFrame]: a 2D array of
            the same type as 'betas', but values transfromed into m-values,
            following the formula: M = log2(Beta / (1 - Beta))
    """
    # First, check whether the input data is correct.
    if type(betas) is np.ndarray:
        if betas.dtype != 'float64':
            raise TypeError("Input is expected to be of dtype('float64')")
    elif type(betas) is pd.core.frame.DataFrame:
        if betas.dtypes.unique()[0] != 'float64':
            raise TypeError("Input is expected to be of dtype('float64')")
    else:
        raise TypeError('Input is expected to be a numpy.ndarray or a pandas.core.frame.DataFrame')
    # Convert betas to m, elementwise (vectorized).
    m_values = np.log2(betas / (1 - betas))
    return m_values


def train_test_split_david(X, y, stratify, test_size = 0.3, n_splits = 1,
                           imputation = 'random', random_state = 1234,
                           max_diff_other = 0.25, max_iter = 1000,
                           verbose = 1):
    """
    Customization of the sklearn.model_selection.train_test_split function.
    This costumization allows stratification by collumns containing NaN
    values (by imputing them) and avoids the problem of the original sklearn
    implementation by which many nested groups for stratification generate
    one sample groups by considering each variable to strarify separately.

    Required:
        import sys
        import numpy as np
        import pandas as pd
        import sklearn.model_selection as model_sel
    Arguments:
        X [pandas.DataFrame]: the array to be splitted into training and test
            splits. Use sample names as index.
        y [pandas.Series]: the target labels corresponding to samples in X.
            It should be in the same order as X. It will be splitted in the
            same way as X. Use sample names as index.
        stratify [pandas.DataFrame / pandas.Series]:
            array-like with the target/s variables to use for stratification.
            If more than one column array is provided, the different columns
            should be placed from most to least important for stratification.
        test_size [float]: the proportion of samples to keep for test
            (validation). Default = 0.3.
        n_splits [int]: number of train/test splits to generate. They will be
            generated using different random seeds. Default = 1.
        imputation [string / False]. The imputation method to use for NaN
            values. Possibilities:
                'most_frequent' for imputation using the most frequent value.
                'random' for random value imputation.
                False for no imputation.
                Default = 'random'.
        random_state [int]: the random seed. Default = 1234.
        max_diff_other [float]: maximum proportion difference for train and
            test in each secondary class. Default = 0.25.
        max_iter [int]: max number of random splits to find a valid one
            for secondary classes. Default = 1000.
        verbose [int]: controls the verbosity of the function messages.
            Levels:
                0: no messages.
                1: medium verbosity level (default).
                2: max verbosity level (for debugging).
    Return:
        [list]: a list of list elements, one for each split generated. Each
            split list contains (by order):
                X_train [pandas.DataFrame]: the training set.
                X_test [pandas.DataFrame]: the test set.
                y_train [pandas.Series]: class labels for training set samples.
                y_test [pandas.Series]: class labels for test set.
                valid_for_stratification_idx [list]: the stratify.columns index
                    possitions of the columns that were actually used for
                    stratification.

    Example:
        my_splits = train_test_split_david(b_vals_response, response_complete,
                                           stratify_df, test_size = 0.3,
                                           n_splits = 5, max_diff_other = 0.1)
    """
    if len(stratify.shape) == 1 or stratify.shape[1] == 1:
        if verbose > 0:
            print('[MSG] Only one variable for stratification detected.')
            print('[MSG] Use sklearn.model_selection.train_test_split')
        return None
    else:
        if verbose > 0:
             print(f'[MSG] {stratify.shape[1]} variables for stratification detected.')
        # It will split first by the first column of stratify. For the 
        # rest of columns, only permutations of samples are allowed if the
        # split by the previous ones is not afected. Then, variables should
        # be ordered by importance (from most to least) in stratify.
        columns_strat = stratify.columns.values
        # Var to store the splits found.
        splits_found = []
        for iteration in range(n_splits):
            if verbose > 0:
                print(f'[MSG] generating split {iteration}.')
                print(f'[MSG] Stratify by column {columns_strat[0]}')
            np.random.seed(random_state + iteration)
            rand_i = np.random.randint(1, 9999)
            # First split.
            X_train, X_test, y_train, y_test = \
                model_sel.train_test_split(X,
                                           y,
                                           test_size = test_size,
                                           random_state = rand_i,
                                           stratify = stratify.iloc[:,0])
            train_samples = X_train.index.values
            test_samples = X_test.index.values
            train_len = len(train_samples)
            test_len = len(test_samples)
            target_vals_list = []  # To store imputed ones.
            # Var to store the columns by which it is possible to 
            # stratify and then should be taken into account 
            # in each new split to check that the new split 
            # does not affect previous columns. Initialize with the first
            # column.
            valid_for_stratification_idx = [0]
            for i in range(1,len(columns_strat)):
                if verbose > 0:
                    print(f'[MSG] Stratification by column {columns_strat[i]}')
                # The current column.
                target_vals = stratify.iloc[:, i].copy()
                # Imputation.
                if sum(target_vals.isnull()) > 0:
                    if imputation is False:
                        sys.exit('[ERROR] You chose no imputation' +
                                 f' but {columns_strat[i]} contain ' +
                                 'NaN values. Consider either impute ' +
                                 'values or discard this column for ' +
                                 'stratification')
                    elif imputation == 'most_frequent':
                        m_freq = target_vals.mode().values[0]
                        target_vals.iloc[target_vals.isnull().values] = m_freq
                    elif imputation == 'random':
                        possible_vals = target_vals.iloc[
                            ~target_vals.isnull().values].unique()
                        new_vals = []
                        for v in range(sum(target_vals.isnull())):
                            np.random.seed(rand_i + v)
                            new_vals.append(np.random.choice(possible_vals))
                        target_vals.iloc[target_vals.isnull().values] = \
                            new_vals
                    else:
                        sys.exit('[ERROR] Imputation argument incorrect ' +
                                 "please use: 'most_frequent', 'random' " + 
                                 'or False')
                target_vals_list.append(target_vals)
                # Check whether the current split satisfy thresholds,
                # i.e, none of the values of target_vals have a 
                # difference in proportion between training and test
                # larger than max_diff_other.
                target_train = target_vals.loc[train_samples]
                target_test = target_vals.loc[test_samples]
                target_train_props = target_train.value_counts() / train_len
                target_test_props = target_test.value_counts() / test_len
                if verbose > 1:
                    print(f'[MSG] Training set proportions for {columns_strat[i]}:')
                    print(target_train_props)
                    print(f'[MSG] Test set proportions for {columns_strat[i]}:')
                    print(target_test_props)
                    print(f'[MSG] Proportion differences for {columns_strat[i]}:')
                    print(abs(target_train_props - target_test_props))
                    print('[MSG] Number of groups with 0 samples in any split:')
                    print(sum(abs(target_train_props - target_test_props).isnull()))
                if any(abs(target_train_props - target_test_props) > 
                       max_diff_other) or (
                   sum(abs(target_train_props -
                           target_test_props).isnull()) > 0):
                    valid_found = False
                    for n in range(max_iter):
                        if verbose > 0:
                            print(f'[MSG] Iterative search for a valid split: iteration {n}')
                        # Try random new splits stratified by the original 
                        # first column and get the first that satisfy 
                        # thresholds, if any.
                        rand_i_c = rand_i + n + i
                        X_train_c, X_test_c, y_train_c, y_test_c = \
                            model_sel.train_test_split(X,
                                                       y,
                                                       test_size = test_size,
                                                       random_state = rand_i_c,
                                                       stratify = stratify.iloc[:,0])
                        train_samples_c = X_train_c.index.values
                        test_samples_c = X_test_c.index.values
                        train_len_c = len(train_samples_c)
                        test_len_c = len(test_samples_c)
                        # Check whether is valid for each of the previous
                        # columns.
                        valid_list = []
                        for p in range(1, i + 1):
                            if ((p in valid_for_stratification_idx) or
                                (p > max(valid_for_stratification_idx))):
                                target_vals_p = target_vals_list[p - 1]
                                target_train_p = target_vals_p.loc[train_samples_c]
                                target_test_p = target_vals_p.loc[test_samples_c]
                                target_train_p_props = target_train_p.value_counts()/train_len_c
                                target_test_p_props = target_test_p.value_counts() / test_len_c
                                if verbose > 1:
                                    print(target_train_p_props)
                                    print(target_test_p_props)
                                    print(abs(target_train_p_props - target_test_p_props))
                                    print(sum(abs(target_train_p_props -
                                                  target_test_p_props).isnull()))
                                if any(abs(target_train_p_props - target_test_p_props) > 
                                       max_diff_other) or (
                                   sum(abs(target_train_p_props -
                                           target_test_p_props).isnull()) > 0):
                                    valid_list.append(False)
                                else:
                                    valid_list.append(True)
                        if all(valid_list):
                            X_train = X_train_c
                            X_test = X_test_c
                            y_train = y_train_c
                            y_test = y_test_c
                            train_samples = train_samples_c
                            test_samples = test_samples_c
                            train_len = train_len_c
                            test_len = test_len_c
                            valid_found = True
                            valid_for_stratification_idx.append(i)
                            if verbose > 0:
                                print(f'[MSG] Valid split for {columns_strat[i]} ' +
                                      f'was found at iteration {n}')
                            break
                    if not valid_found:
                        if verbose > 0:
                            print('[MSG] A valid splitting for '
                                  f'{columns_strat[i]} was not found.')
                else:
                    # The current split is already good for this new col.
                    if verbose > 0:
                        print(f'[MSG] Iterative search not necessary for {columns_strat[i]}.')
                    valid_for_stratification_idx.append(i)
            # Return the split.
            new_split = [X_train, X_test, y_train, y_test,
                         valid_for_stratification_idx]
            splits_found.append(new_split)
        return splits_found

def hard_sample_detector(estimator, X, y, groups=None, cv=None, n_jobs=None,
                         verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                         return_train_pred=False, return_estimator=False,
                         error_fit=np.nan):
    """Evaluate sample predictability using cross-validation.

    In many real world problems, the available samples are not always usable
    for the model training. These samples are characterized for being very
    hard to correctly classify. It could be due to many reasons: bad sample
    annotation, extreme outliers, etc. By including this kind of samples the
    resulting model can be biased to specifically target these hard samples and
    making very difficult the creation of a generalizable model. To fight
    against this and to have a better idea about the samples behaviour, this
    function will perform several rounds of CV testing each sample many times
    and reporting the number of successes of each. Samples with very low
    successes could be considered to be checked/discarded.

    Note: this function is only usable for supervised classification problems.

    Note 2: while this could be implemented as a transformer to be included
    in Pipeline, I do not do it for two main reasons: a) adding an additional
    CV procedure to a Pipeline could be very computational intensive; b) the
    inclusion in a pipeline needs some mechanism for automatic removal of the
    hard samples, this could be a source of potential problems. I made this
    function to provide means to investigate hard samples, but automatically
    removing them is a bit drastic.

    Note 3: based on sklearn.model_selection._validation cross_validate.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    return_train_pred : bool, default=False
        Whether to include train predictions.
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
    error_fit : 'raise' or numeric, default=np.nan
        Value to assign to the predictions if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    Returns
    -------
    ret : dict with the following possible keys (see ``parameters``):
        fit_time : list of int
            Time used to fit the estimators, in seconds, for the train set in
            each of the CV folds.
        pred_time : list of int
            Time used to make predictions, in seconds, for the train set in
            each of the CV folds.
        estimator : list of estimator
            The estimator objects for each cv split. Only available if
            ``return_estimator`` is set to True.
        test_pred : list of tuple
            List of tuples, each with two arrays, the firs indicating the
            sample index position and the second if it was correctly (True)
            or not (False) classified. One element of the list for each fold
            of the CV. Results of the test set of each CV fold.
        train_pred : list of tuple
            List of tuples, each with two arrays, the firs indicating the
            sample index position and the second if it was correctly (True)
            or not (False) classified. One element of the list for each fold
            of the CV. Results of the training set of each of the CV fold.
        test_success : array of shape (len(y), )
            An array in wich each of the positions indicates the number
            of successful predictions in the test set predictions. With this
            array one can study the hard samples looking at samples with
            low successes.
        train_success : array fo shape (len(y), )
            An array un wich each of the positions indicates the number
            of successful predictions in the training set predictions. This
            is generally less informative, but can serve to investigate the
            capacity of the estimator to overfit to hard samples.
    Examples
    --------

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_report)(
            clone(estimator), X, y, train, test, verbose, None,
            fit_params, return_train_pred=return_train_pred,
            return_times=True, return_estimator=return_estimator,
            error_fit=error_fit)
        for train, test in cv.split(X, y, groups))
    ret = {}
    ret['fit_time'] = [r["fit_time"] for r in results]
    ret['pred_time'] = [r["pred_time"] for r in results]

    if return_estimator:
        ret['estimator'] = [r["estimator"] for r in results]

    ret['test_pred'] = [r["test_pred"] for r in results]

    y_test_acc = np.zeros(len(y))
    for idx, acc in ret['test_pred']:
        y_test_acc[idx] += acc
    ret['test_success'] = y_test_acc

    if return_train_pred:
        ret['train_pred'] = [r["train_pred"] for r in results]

        y_train_acc = np.zeros(len(y))
        for idx, acc in ret['train_pred']:
            y_train_acc[idx] += acc
        ret['train_success'] = y_train_acc

    return ret

def _fit_and_report(estimator, X, y, train, test, verbose,
                    parameters, fit_params, return_train_pred=False,
                    return_parameters=False,
                    return_n_test_samples=False, return_times=False,
                    return_estimator=False, split_progress=None,
                    candidate_progress=None, error_fit=np.nan):
    """Fit estimator and compute non-correctly predicted samples for a given
    dataset split.

    Note: based on sklearn.model_selection._validation _fit_and_score function.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_fit : 'raise' or numeric, default=np.nan
        Value to assign to the result if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_pred : bool, default=False
        Compute and return predictions on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    Returns
    -------
    result : dict with the following attributes
        train_pred : tuple of arrays
            Tuple of arrays. First element is the indexes (ints) of the samples
            evaluated, second element is an array of bools where True
            means correctly classified and False incorrectly. Contains
            the predictions results of the training set.
        test_pred : tuple of arrays
            Tuple of arrays. First element is the indexes (ints) of the samples
            evaluated, second element is an array of bools where True
            means correctly classified and False incorrectly. Contains
            the predictions results of the test set.
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        pred_time : float
            Time spent for predictions in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
    """
    if not isinstance(error_fit, numbers.Number) and error_fit != 'raise':
        raise ValueError(
            "error_fit must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (f"; {candidate_progress[0]+1}/"
                             f"{candidate_progress[1]}")

    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = (', '.join(f'{k}={parameters[k]}'
                                    for k in sorted_keys))
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        pred_time = 0.0
        if error_fit == 'raise':
            raise
        elif isinstance(error_fit, numbers.Number):
            test_pred = error_fit
            if return_train_pred:
                train_pred = error_fit
            warnings.warn("Estimator fit failed. The fot on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_fit, format_exc()),
                          FitFailedWarning)
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        test_pred = (test,
                     _pred_result_per_sample(estimator, X_test, y_test)
                    )

        pred_time = time.time() - start_time - fit_time
        if return_train_pred:
            train_pred = (train,
                          _pred_result_per_sample(
                            estimator, X_train, y_train)
                         )

    if verbose > 1:
        total_time = pred_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_pred"] = test_pred
    if return_train_pred:
        result["train_pred"] = train_pred
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["pred_time"] = pred_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result

def _pred_result_per_sample(estimator, X, y):
    """Predicts with an already fitted estimator and returns the fails (False)
    and successes (True) for each of the samples, compared with the real labels
    (``y``).

    Parameters
    ----------
    estimator : estimator object implementing 'predict'
        The object to use to predict the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    Return
    ------
    s : array of bools of shape (len(y))
        Bool array indicating the correctly predicted ``y`` labels (True) or
        incorrect (False).
    """
    #check_is_fitted(estimator) # It fails with some Pipelines
    y_pred = estimator.predict(X)
    s = y_pred == y
    return s

