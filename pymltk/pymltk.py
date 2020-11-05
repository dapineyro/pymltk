#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python Machine Learning Toolkit
"""

#===============================================================================
# Script Name: pymltk.py
#
# Author: David Piñeyro
# Contact info: dpineyro@carrerasresearch.org
# Date: 2020-08-04
# Version: 0.0.4
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
#   along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Description:
#   The present script is meant to be a collection of objects and functions to
#   be used for machine learning in Python. It is probable that it eventually
#   will become a pyhon module/package. For the moment, import it to your
#   scripts.
#
# Usage:
#   import pymltk as mlt
#
# Requisites:
#   The following packages are required for the module functions. Check each 
#   function docstring to know specific requirements.
#       sys
#       argparse
#       numpy
#       pandas
#       sklearn
#       scipy
#       
#===============================================================================

#===============================================================================
# IMPORTS
import sys
import numpy as np
import pandas as pd
import sklearn.model_selection as model_sel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_X_y, check_array
from scipy.stats import spearmanr, f_oneway

#===============================================================================
# GLOBAL VARIABLES
SCRIPT_DESCRIPTION = """
Python ML toolkit
Version: 0.0.2
Author: David Piñeyro
Date: 2020-08-04
License: GPL-3
"""
__version__ = '0.0.4'

GLOBAL_ITERATOR = 0

#===============================================================================
# FUNCTIONS
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

#===============================================================================
# CLASSES
class MethylData:
    """Convenience class to handle methylation data fo ML.

    Convenience class to handle all the required ML data and opperations from
    methylation arrays.

    Parameters
    ----------
    betas : string
        Path to betas.csv file. It should be a comma sepparated file.
        Samples as columns and CpGs as rows. It should have a header with
        sample names and the first column, which correponds to CpG names,
        may be unnamed. All values are expected to be floats between 0 and 1 or
        the NA string.
    sample_sheet : string
        Path to sample_sheet.csv file. It should be a comma sepparated file,
        with samples as rows and features as columns. It should contain at
        least two columns:
            'Sample_Name' and class_label.
    class_label : string
        The column name of the column in the sample_sheet to be used a
        classification label.
    skip_ss : int, default = 0
        Number of rows to skip from the sample_sheet. This option is
        hard-coded to deal with the typical Illumina sample sheet in which
        their first 7 rows has to be skipped.
    Attributes
    ----------
    betas_original_ : pandas.DataFrame
        The original input betas passed when the object was created.
    betas_ : pandas.DataFrame
        The original input betas but processed by the load_data
        method. Samples as rows and features (CpGs) as columns. Row index
        will be named 'Sample_Name' and column index will be named 'Probes'.
        All samples without class_label value will be removed.
    m_values_ : pandas.DataFrame
        The same as betas_ but with all the values converted to m_values
        using my pymltk.betas2m function.
    sample_sheet_original_ : pandas.DataFrame
        The original input sample_sheet passed when the object was created.
    sample_sheet_ : pandas.DataFrame
        The original input sample_sheet but processed by the load_data
        method. Samples as rows and features as columns. Its the same as
        the original, but with each sample with np.nan in class_label
        eliminated.
    Y_ : pandas.Series
        The target variable (classification variable) from the
        sample_sheet[class_label].
    class_label_ : string
        The column name of the column in the sample_sheet to be used a
        classification label.
    island_betas_ : pandas.DataFrame
        Median metilation beta value for each CpG island listed in `manifest`.
        Samples as rows and CpG_island as columns. Row index will be named
        'Sample_Name' and column index will be named 'Islands'.
    distance_group_betas_ : pandas.DataFrame
        Median metilation beta value for each CpG distance group.
        Samples as rows and CpG_island as columns. Row index will be named
        'Sample_Name' and column index will be named 'DG'.
    island_m_ : pandas.DataFrame
        Median metilation m value for each CpG island listed in `manifest`.
        Samples as rows and CpG_island as columns. Row index will be named
        'Sample_Name' and column index will be named 'Islands'.
    distance_group_m_ : pandas.DataFrame
        Median metilation m value for each CpG distance group.
        Samples as rows and CpG_island as columns. Row index will be named
        'Sample_Name' and column index will be named 'DG'.
    manifest_ : pandas.DataFrame
        Original DataFrame from Illumina documentation with info about
        probes.
    manifest_islands_ : pandas.DataFrame
        Reduced version of the `manifest_`, containing only info about
        probes in CpG islands.
    manifest_distance_group_ :pandas.DataFrame
        Reduced version of the `manifest_`, containing only info about
        groups of probes grouped by distance.
    """
    def __init__(self, betas, sample_sheet, class_label, skip_ss = 0):
        self.class_label_ = class_label
        self.load_data(betas, sample_sheet, class_label, skip_ss)

    def load_data(self, betas, sample_sheet, class_label, skip_ss):
        """Collects data while checking its correctness.

        Parameters
        ----------
        betas : string
            Path to betas.csv file. It should be a comma sepparated file.
            Samples as columns and CpGs as rows. It should have a header with
            sample names and the first column, which correponds to CpG names,
            may be unnamed. All values are expected to be floats between 0 and 1 or
            the NA string.
        Sample_sheet : string
            Path to sample_sheet.csv file. It should be a comma sepparated file,
            with samples as rows and features as columns. It should contain at
            least two columns:
                'Sample_Name' and class_label.
        class_label : string
            The column name of the column in the sample_sheet to be used a
            classification label.
        skip_ss : int
            Number of rows to skip from the sample_sheet. This option is
            hard-coded to deal with the typical Illumina sample sheet in which
            their first 7 rows has to be skipped.
        Set attributes
        --------------
        betas_original_ : pandas.DataFrame
        betas_ : pandas.DataFrame
        m_values_ :pandas.DataFrame
        sample_sheet_original_ : pandas.DataFrame
        sample_sheet_ : pandas.DataFrame
        Y_ : pandas.Series
        """
        print('[MSG] Loading data...')
        try:
            betas = pd.read_csv(betas)
            self.betas_original_ = betas.copy(deep = True)
            # Change first column name and assign that to rownames.
            betas_colnames = betas.columns.values
            betas_colnames[0] = 'Probes'
            betas.columns = betas_colnames
            betas.index = betas['Probes']
            # Remove the first column.
            betas.drop(['Probes'], axis = 1, inplace = True)
            betas.columns.names = ['Sample_Name']
            betas = betas.transpose()
        except OSError as err:
            print('[OSError] {}'.format(err))
            sys.exit()

        try:
            sample_sheet = pd.read_csv(sample_sheet, skiprows = skip_ss)
            self.sample_sheet_original_ = sample_sheet.copy(deep = True)
            # Add rownames as Sample_Name column.
            sample_sheet.index = sample_sheet['Sample_Name']
        except OSError as err:
            print('[OSError] {}'.format(err))
            sys.exit()

        # Check the presence of class label in sample_sheet.
        if class_label not in sample_sheet.columns.values:
            print(f'[ERROR] Column {class_label} is ' +
                  f'not present in {sample_sheet}')
            sys.exit()

        # Remove rows with NaN in class_label.
        nan_idx = pd.isnull(sample_sheet[class_label])
        sample_sheet_processed = sample_sheet.loc[~ nan_idx, :]
        betas_processed = betas.loc[sample_sheet_processed['Sample_Name'],
                                    :]
        # Convert to m_values.
        m_vals = betas2m(betas_processed)
        self.m_values_ = pd.DataFrame(data = m_vals,
                                     index = betas_processed.index,
                                     columns = betas_processed.columns)

        print('[MSG] Data loaded successfully')
        print('[MSG] Original betas shape:', betas.shape)
        print('[MSG] Original sample_sheet shape:', sample_sheet.shape)
        print('[MSG] Processed betas shape:', betas_processed.shape)
        print('[MSG] Processed m_values shape:', self.m_values_.shape)
        print('[MSG] Processed sample_sheet shape:', sample_sheet_processed.shape)

        self.betas_ = betas_processed
        self.sample_sheet_ = sample_sheet_processed
        self.Y_ = sample_sheet_processed[class_label]
        return self

    def group_cpgs_by_island(self, manifest_path, skiprows = 7,
                             verbose = False):
        """Group CpGs by their island median beta value.

        Based on the infomation of `manifest` from `island_col`, this function
        groups all the CpGs belonging to a given CpG island, computing their
        median beta-value and generating a new pandas.DataFrame with the same
        samples as `self.betas_` but with CpG island mean beta-values as
        features (columns). An equivalent m_values pandas.DataFrame will be
        also generated.

        Parameters
        ----------
        manifest_path : string
            String with the path to the manifest file (usually from Illumina
            documentation) with all the information regarding aray probes.
            Probes as rows and probe features as columns. At least the
            following columns should exist:
            'Name', 'CHR', 'MAPINFO' in addition to `island_col`.
        skiprows : int, default = 7
            Number of rows to be skipped from the begining. Illumina usually
            puts 7 lines with additional info before the table header and
            should be skipped.
        verbose : bool, default = False
            Whether to print messages or not.

        Set attributes
        --------------
        island_betas_ : pandas.DataFrame
        island_m_ : pandas.DataFrame
        manifest_ : pandas.DataFrame
        manifest_islands_ : pandas.DataFrame
        """
        # Required colnames from illumina manifest file.
        name_col = 'Name'
        chr_col = 'CHR'
        pos_col = 'MAPINFO'
        island_col = 'UCSC_CpG_Islands_Name'
        relation_col = 'Relation_to_UCSC_CpG_Island'
        required_cols = [name_col, chr_col, pos_col, island_col, relation_col]
        # Processing manifest.
        annot = pd.read_csv(manifest_path, skiprows = skiprows,
                            low_memory = False)
        if verbose:
            if hasattr(self, 'manifest_'):
                print('[MSG] The attribute \'manifest_\' will be updated.')
        self.manifest_ = annot.copy(deep = True)
        annot_islands = annot.loc[~ annot[island_col].isna(), required_cols]
        annot_islands_s = annot_islands.loc[
            annot_islands[relation_col] == 'Island', :]
        annot_islands_sorted = annot_islands_s.sort_values(by = [island_col])
        annot_short = annot_islands_sorted.loc[:, [island_col]]
        annot_short.index = annot_islands_sorted[name_col]
        self.manifest_islands_ = annot_short.copy(deep = True)
        if verbose:
            print(f'[MSG] {annot_short.shape[0]} probes to be grouped to ' +
                  f'{annot_short[island_col].unique().shape[0]} ' +
                  'CpG islands in manifest.')
        # Merge with betas sample by sample.
        betas_t = self.betas_.T
        islands = betas_t.apply(lambda x: dict(
            pd.merge(x, annot_short, left_index = True,
                     right_index = True).groupby(
                         by = island_col).median()),
            axis = 0)
        d_to_df = {k: v for i in islands for k, v in i.items()}
        island_betas = pd.DataFrame(d_to_df).T
        island_betas.index.name = 'Sample_Name'
        island_betas.columns.name = 'Islands'
        if verbose:
            print(f'[MSG] A total of {island_betas.shape[1]} islands ' +
                  'collected as new features in island_betas_')
        self.island_betas_ = island_betas
        self.island_m_ = betas2m(island_betas)
        return self

    def group_cpgs_by_distance(self, manifest_path, dist_thr = 1000,
                               skiprows = 7, verbose = False):
        """Group CpGs by their relative distance.

        Based on the infomation of `manifest`, this function groups all the
        CpGs by their relative distance, computing their median beta-value
        and generating a new pandas.DataFrame with the same samples as
        `self.betas_` but with grouped CpGs mean beta-values as features
        (columns). An equivalent m_values pandas.DataFrame will be also
        generated.

        Grouping algorithm:
            - Separate probes by chromosome.
            - Sort probes by `MAPINFO` col.
            - Calculate the conscutive distances using pd.DataFrame.diff().
            - Group all that probes that can be connected by distances not
              grater than `dist_thr` bases.

        Parameters
        ----------
        manifest_path : string
            String with the path to the manifest file (usually from Illumina
            documentation) with all the information regarding aray probes.
            Probes as rows and probe features as columns. At least the
            following columns should exist:
            'Name', 'CHR', 'MAPINFO'.
        dist_thr : int
            Maximum distance (in bp) between two consecutive probes to be
            considered as the same group.
        skiprows : int, default = 7
            Number of rows to be skipped from the begining. Illumina usually
            puts 7 lines with additional info before the table header and
            should be skipped.
        verbose : bool, default = False
            Whether to print messages or not.

        Set attributes
        --------------
        distance_group_betas_ : pandas.DataFrame
        distance_group_m_ : pandas.DataFrame
        manifest_ : pandas.DataFrame
        manifest_distance_group_ : pandas.DataFrame
        """
        # Required colnames from illumina manifest file.
        name_col = 'Name'
        chr_col = 'CHR'
        pos_col = 'MAPINFO'
        required_cols = [name_col, chr_col, pos_col]
        # Processing manifest.
        annot = pd.read_csv(manifest_path, skiprows = skiprows,
                            low_memory = False)
        if verbose:
            if hasattr(self, 'manifest_'):
                print('[MSG] The attribute \'manifest_\' will be updated.')
        self.manifest_ = annot.copy(deep = True)
        annot_chr = annot.loc[~ annot[chr_col].isna(), required_cols]
        annot_by_chr = annot_chr.groupby(by = chr_col)
        annot_by_chr_diff = [pd.concat(
                              [v.sort_values(by = pos_col),
                               v.sort_values(by = pos_col)[pos_col].diff()],
                              axis = 1)
                             for k,v in annot_by_chr]
        # Rename distance col.
        for df in annot_by_chr_diff:
            df.columns.values[3] = 'Distance'
        # Group by distance.
        for df in annot_by_chr_diff:
            chrom = df.iloc[0, 1]
            global GLOBAL_ITERATOR
            GLOBAL_ITERATOR = 1
            dg = df['Distance'].apply(self._dist_gr_name,
                                      chrom = chrom,
                                      dist_thr = dist_thr)
            df['DG'] = dg
        # Concatenate by chr.
        annot_groups = pd.concat(annot_by_chr_diff)
        self.manifest_distance_group_ = annot_groups.copy(deep = True)
        if verbose:
            dg = 'DG'
            print(f'[MSG] {annot_groups.shape[0]} probes to be grouped to ' +
                  f'{annot_groups[dg].unique().shape[0]} ' +
                  'distance groups.')
        annot_short = annot_groups.loc[:, ['DG']]
        annot_short.index = annot_groups[name_col]
        # Merge with betas sample by sample.
        betas_t = self.betas_.T
        d_groups = betas_t.apply(lambda x: dict(
            pd.merge(x, annot_short, left_index = True,
                     right_index = True).groupby(
                         by = 'DG').median()),
            axis = 0)
        d_to_df = {k: v for i in d_groups for k, v in i.items()}
        distance_group_betas = pd.DataFrame(d_to_df).T
        distance_group_betas.index.name = 'Sample_Name'
        distance_group_betas.columns.name = 'DG'
        if verbose:
            print(f'[MSG] A total of {distance_group_betas.shape[1]} groups ' +
                  'collected as new features in distance_group_betas_')
        self.distance_group_betas_ = distance_group_betas
        self.distance_group_m_ = betas2m(distance_group_betas)
        return self

    def _dist_gr_name(self, x, chrom, dist_thr):
        """Distance group name assignation.

        This funtion is intented to be used internally by
        group_cpgs_by_distance method. Assign a name for the distance
        group based on its chromosome and order.

        Parameters
        ----------
        x : int, float
            The distance with to previous probe.
        chrom : string
            The chromosome in '1', '2', 'X', 'Y', 'M' notation.
        dist_thr : int
            The threshold value of `x` by which a change of group name
            happens.
        Returns
        ------
        dg_val : string
            The distance group name for the given probe.
        """
        global GLOBAL_ITERATOR
        if np.isnan(x) or x <= dist_thr:
            dg_val = 'dg_' + str(chrom) + '_' + str(GLOBAL_ITERATOR)
            return dg_val
        else:
            GLOBAL_ITERATOR += 1
            dg_val = 'dg_' + str(chrom) + '_' + str(GLOBAL_ITERATOR)
            return dg_val

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
        An array (np.Series) of bool with len == n_features_original_ where
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
            raise ValueError('Shape of input is different from what was seen'
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
        An array (np.Series) of bool with len == n_features_original_ where
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
            x_groups = np.array([x_i[y == t] for t in targets])
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
            raise ValueError('Shape of input is different from what was seen'
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

