"""
Collection of useful utilities to preprocess data.
"""

import sys
import numpy as np
import pandas as pd
import sklearn.model_selection as model_sel

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


def train_test_split_david(X, Y, stratify, test_size = 0.3, n_splits = 1,
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
        Y [pandas.Series]: the target labels corresponding to samples in X.
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
                Y_train [pandas.Series]: class labels for training set samples.
                Y_test [pandas.Series]: class labels for test set.
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
            X_train, X_test, Y_train, Y_test = \
                model_sel.train_test_split(X,
                                           Y,
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
                        X_train_c, X_test_c, Y_train_c, Y_test_c = \
                            model_sel.train_test_split(X,
                                                       Y,
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
                            Y_train = Y_train_c
                            Y_test = Y_test_c
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
            new_split = [X_train, X_test, Y_train, Y_test,
                         valid_for_stratification_idx]
            splits_found.append(new_split)
        return splits_found
