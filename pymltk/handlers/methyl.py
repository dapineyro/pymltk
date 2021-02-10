"""
Classes to handle methylation data.
"""

import sys
import numpy as np
import pandas as pd

from ..utils.preprocess import betas2m


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
        # Select betas present in the processed sample_sheet, and put them
        # in the same order.
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

