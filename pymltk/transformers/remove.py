"""
Collection of transformers used to remove features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_X_y, check_array
from scipy.stats import spearmanr, f_oneway
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

class RemoveFeaturesByNaN(TransformerMixin, BaseEstimator):
    """Remove features based on proporion of NaN values.

    This transformer returns the data matrix without all the features having
    a proportion of NaN higher than max_nan_prop.

    Parameters
    ----------
    max_nan_prop : float, default=0.1
        Maximum proportion of np.NaN values per feature allowed.
    verbose: bool, default = False
        Controls the verbosity level. Default = no MGS.
    Attributes
    ----------
    max_nan_prop : float, default=0.1
        Maximum proportion of np.NaN values per feature allowed.
    verbose: bool, default = False
        Controls the verbosity level. Default = no MGS.
    n_features_original_ : int
        The number of features of the data passed to :meth:`fit`.
    n_features_transformed_ : int
        The number of features of the transformed data.
    features_to_keep_ : 1D array of bool.
        An array (np.array) of bool with len == n_features_original_ where
        True means a conserved feature in the transformed data.
    """
    def __init__(self, max_nan_prop = 0.1, verbose = False):
        self.max_nan_prop = max_nan_prop
        self.verbose = verbose

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True, force_all_finite = False)

        self.n_features_original_ = X.shape[1]

        if self.verbose:
            print('[MSG] Fitting to select features by prop of failed samples ' +
                  f'over {self.max_nan_prop}')

        nan_per_feature = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0,
                                          X)
        not_failed_features = nan_per_feature < round(
            X.shape[0] * self.max_nan_prop, 0)
        self.features_to_keep_ = not_failed_features
        self.n_features_transformed_ = np.sum(not_failed_features)

        if self.verbose:
            if self.n_features_transformed_ == self.n_features_original_:
                print('[MSG] No failed features found with a prop of failed ' +
                      f'samples over {self.max_nan_prop}')
            else:
                print('[MSG] Number of failed features to be removed: ' +
                      f'{np.sum(~ not_failed_features)}')

        # Return the transformer.
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_original_')

        # Input validation
        X = check_array(X, accept_sparse=True, force_all_finite = False)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_original_:
            raise ValueError('Shape of input is different from what was seen' +
                             'in `fit`')

        if self.verbose:
            print('[MSG] Removing features by prop of failed samples over ' +
                  f'{self.max_nan_prop}')
        if self.n_features_transformed_ == self.n_features_original_:
            if self.verbose:
                print('[MSG] No failed features found with a prop of failed ' +
                      f'samples over {self.max_nan_prop}')
            return X
        else:
            if self.verbose:
                print('[MSG] Number of failed features removed: ' +
                      f'{np.sum(~ self.features_to_keep_)}')
            return X[:, self.features_to_keep_]

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected
        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.
        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True if its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_original_')
        mask = self.features_to_keep_
        return mask if not indices else np.where(mask)[0]

class RemoveCorrelatedFeatures(TransformerMixin, BaseEstimator):
    """Remove features based on their correlation.

    This transformer returns the data matrix with only one of each
    correlated feature. For each group of correlated features, all
    but one are discarded, following the specified method. It can
    be used only for classification.

    Parameters
    ----------
    select_by : string, default='anova'
        The method to select between correlated features. Possibilities
        are 'random' and 'anova'.
    cor_thr : float, default=0.9
        Correlation value above wich two or more features will be
        considered correlated. Absolute values will be taken into
        account, so -0.9 or 0.9 will have the same effect. Expected
        values ranges from 0 to 1.
    method : string, default='pearson'
        Statistical method to assess correlation. Available methods are
        'pearson' and 'spearman'.
    random_seed : int, None, default = None.
        Random seed used in np.random.seed to get reproducible results.
    verbose: bool, default = False
        Controls the verbosity level. Default = no MGS.
    Attributes
    ----------
    cor_matrix_ : numpy.ndarray
        The features correlation matrix.
    n_features_original_ : int
        The number of features of the data passed to :meth:`fit`.
    n_features_transformed_ : int
        The number of features of the transformed data.
    features_to_keep_ : 1D array of bool
        An array (np.array) of bool with len == n_features_original_ where
        True means a conserved feature in the transformed data.
    """
    def __init__(self, select_by = 'anova', cor_thr = 0.9,
                 method = 'pearson', random_seed = None, verbose = False):
        self.select_by = select_by
        self.cor_thr = cor_thr
        self.method = method
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : 1D array, shape (n_samples, )
            The target variable. It is required when select_by is not 'random'.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, force_all_finite = False)

        self.n_features_original_ = X.shape[1]

        if self.verbose:
            print('[MSG] Fitting to select features by correlation by ' +
                  f'{self.method} method and with a correlation coefficient ' +
                  f'threshold of {self.cor_thr}')

        if self.method == 'spearman':
            cor_mat = spearmanr(X)[0]
        elif self.method == 'pearson':
            cor_mat = np.corrcoef(X, rowvar = False)
        else:
            raise ValueError('Parameter method is not correctly specified ' +
                             'please use either \'pearson\' or \'spearman\'.')
        self.cor_matrix_ = cor_mat

        # Generate a bool mask matrix with True when there is a correlation
        # above the thr. Discard the diagonal.
        cor_mask = np.triu(np.abs(cor_mat) > self.cor_thr, k = 1)

        ### Algorithm to decide features to discard.
        cor_idx = np.where(cor_mask) # Tuple where [0] rows and [1] cols.
        # As the cor_mask is the upper part of the cor_mat, the algorithm
        # is row-wise.
        row_idx = cor_idx[0]
        col_idx = cor_idx[1]
        interacting_features = np.unique(row_idx)
        # Feature groups by rows of cor_mat
        f_groups = [np.append(f, col_idx[np.where(row_idx == f)])
                        for f in interacting_features]
        features_to_keep = np.full(self.n_features_original_, True)
        for g in f_groups:
            # Check whether the current first feature was already discarded.
            if features_to_keep[g[0]]:
                if self.select_by == 'random':
                    # Select one of the correlated features randomly.
                    np.random.seed(self.random_seed)
                    chosen = np.random.choice(g, size = 1)
                    to_discard = g[g != chosen]
                    features_to_keep[to_discard] = False
                elif self.select_by == 'anova':
                    # Select one of the correlated features by ANOVA.
                    X_group = X[:, g]
                    pvals = self._f_oneway_wrapper(X_group, y)
                    chosen = g[np.where(pvals == pvals.min())]
                    chosen = chosen[0]  # in case there is more than one.
                    to_discard = g[g != chosen]
                    features_to_keep[to_discard] = False
                else:
                    raise ValueError('Parameter select_by is not correctly ' +
                                     'specified. Please use \'random\' or ' +
                                     '\'lm\'.')

        self.features_to_keep_ = features_to_keep
        self.n_features_transformed_ = np.sum(features_to_keep)

        if self.verbose:
            if self.n_features_transformed_ == self.n_features_original_:
                print('[MSG] No correlated features found.')
            else:
                print('[MSG] Number of correlated features to be removed: ' +
                      f'{np.sum(~ features_to_keep)}')

        # Return the transformer.
        return self

    def _f_oneway_wrapper(self, X, y):
        """Wrapper to scipy.stats.f_oneway from X and y arrays.

        Parameters
        ----------
        X : numpy.array, shape(n_samples, >1 features).
            The sliced X array with the features selection to test.
        y : 1D numpy.array, shape(n_samples, ).
            The target variable.
        Returns
        -------
        f_pvalues : 1D numpy.array of floats.
            The list of f_oneway results pvalues, one for each column in X.
        """
        targets = np.unique(y)
        f_pvalues = []
        for i in range(X.shape[1]):
            x_i = X[:,i]
            x_groups = np.array([x_i[y == t] for t in targets], dtype = object)
            r = f_oneway(*x_groups)
            f_pvalues.append(r[1])
        return np.array(f_pvalues)

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_original_')

        # Input validation
        X = check_array(X, accept_sparse=True, force_all_finite = False)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_original_:
            raise ValueError('Shape of input is different from what was seen' +
                             'in `fit`')

        if self.verbose:
            print('[MSG] Removing features by correlation over ' +
                  f'{self.cor_thr} by {self.method} in the training dataset.')
        if self.n_features_transformed_ == self.n_features_original_:
            if self.verbose:
                print('[MSG] No correlated features found.')
            return X
        else:
            if self.verbose:
                print('[MSG] Number of correlated features removed: ' +
                      f'{np.sum(~ self.features_to_keep_)}')
            return X[:, self.features_to_keep_]

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected
        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.
        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True if its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_original_')
        mask = self.features_to_keep_
        return mask if not indices else np.where(mask)[0]

class LimmaFS(TransformerMixin, BaseEstimator):
    """Select features based on limma.

    This transformer statistically test each class against the rest
    and selects the best features for each contrast up to a specified
    limit. To statistically assess the contrasts it uses the R
    implementation of the limma (linear models for microarrays) package.
    If there are more than two classes to predict, each contrast will
    be implemented as one vs the rest.
    Note that to use this function, R and limma package should be
    available in the system. It can be used only for classification. It
    will not accept input data (X) with NaN values (those must be imputed
    or removed before).

    Parameters
    ----------
    k : int
        The maximum number of features to be selected. Those will be
        evenly selected from the top statistically significant in each
        contrast.
    remove_dup : bool, default=True
        Whether to remove duplicate features from the selection. This
        will be only relevant for more than 2 classes datasets. In these
        cases, a round(k/n_classes, 0) features will be selected per each
        of the one vs the rest contasts. It is possible to select the
        same feature in more than one contrast, so `remove_dup` = True
        avoids this behaviour.
    to_m_vals : bool, default=False
        This selection method was originally designed to be used with
        methylation data. This data comes usually in the form of a
        data matrix of beta-values. Those beta-values are not well
        suited for statistical assessment with limma. Then, a data
        transformation to m-values (log2(betas/(1-betas))) is usually
        performed. This parameter, when is set to False (default) do not
        perform this transformation. Set to True when using only beta values.
        Take into account that when True, data must not contain 0 or 1
        values.
    verbose: bool, default = False
        Controls the verbosity level. Default = no MSG.
    Attributes
    ----------
    n_features_original_ : int
        The number of features of the data passed to :meth:`fit`.
    n_features_transformed_ : int
        The number of features of the transformed data.
    features_to_keep_ : 1D array of bool
        An array (np.array) of bool with len == n_features_original_ where
        True means a conserved feature in the transformed data.
    """
    def __init__(self, k, remove_dup = True, to_m_vals = False,
                 verbose = False):
        self.k = k
        self.remove_dup = remove_dup
        self.to_m_vals = to_m_vals
        self.verbose = verbose

    def fit(self, X, y):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples. No NaN values are allowed.
        y : 1D array, shape (n_samples, )
            The target variable.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, force_all_finite = False)

        self.n_features_original_ = X.shape[1]

        if self.verbose:
            print(f'[MSG] Fitting to select {self.k} features using limma')

        # Check for NAs and raise an error when found.
        if np.any(np.isnan(X)):
            raise ValueError('X (input data set) contains NaN values, ' +
                             'please, remove or impute them before.')

        y_pandas = pd.Series(y)
        # numpy arrays do not have rownames and we need them for the R
        # funtion. Just create dummy ones.
        sample_names = pd.Series(np.arange(0, X.shape[0])).astype(str)
        # Load some useful R functions.
        factor_R = ro.r['factor']
        source_R = ro.r['source']

        with localconverter(ro.default_converter + pandas2ri.converter):
            targets_r = ro.conversion.py2rpy(y_pandas)
        targets_r = factor_R(targets_r)

        with localconverter(ro.default_converter + pandas2ri.converter):
            sample_names_r = ro.conversion.py2rpy(sample_names)

        #b_vals_2_r_np = b_vals_2_r.values
        numpy2ri.activate()
        ro.globalenv['X_r'] = X
        numpy2ri.deactivate()

        X_r = ro.r['X_r']

        source_R('../Rcode/limma_fs.R')
        limma_fs = ro.globalenv['limma_fs']

        ro.globalenv['are_m_vals_r'] = not self.to_m_vals
        are_m_vals_r = ro.r['are_m_vals_r']

        ro.globalenv['remove_dup_r'] = self.remove_dup
        remove_dup_r = ro.r['remove_dup_r']

        ro.globalenv['k_r'] = self.k
        k_r = ro.r['k_r']

        sel_r = limma_fs(X_r, sample_names_r, targets_r, k_r, remove_dup_r,
                         are_m_vals_r)
        sel_idx= pd.Series(sel_r).values.astype(int)
        # Remember to subtract 1 as python numbering starts at 0 and at 1 in R.
        sel_idx -= 1
        # Create a bool mask.
        sel_mask = np.array([True if i in sel_idx else False for i in range(0, X.shape[1])])

        self.features_to_keep_ = sel_mask
        self.n_features_transformed_ = np.sum(sel_mask)

        if self.verbose:
            if self.n_features_transformed_ == self.n_features_original_:
                print('[MSG] All original features were selected.')
            else:
                print('[MSG] Number of features to be removed: ' +
                      f'{np.sum(~ features_to_keep)}')

        # Return the transformer.
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_original_')

        # Input validation
        X = check_array(X, accept_sparse=True, force_all_finite = False)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_original_:
            raise ValueError('Shape of input is different from what was seen' +
                             'in `fit`')

        if self.verbose:
            print('[MSG] Selecting features by LimmaFS')
        if self.n_features_transformed_ == self.n_features_original_:
            if self.verbose:
                print('[MSG] All original features were selected.')
            return X
        else:
            if self.verbose:
                print('[MSG] Number of features to be removed: ' +
                      f'{np.sum(~ self.features_to_keep)}')
            return X[:, self.features_to_keep_]

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected
        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.
        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True if its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_original_')
        mask = self.features_to_keep_
        return mask if not indices else np.where(mask)[0]
