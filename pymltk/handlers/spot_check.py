"""
Classes to add or wrapp sklearn functionality
"""

import numpy as np
import pandas as pd

def check_list_of_tuples(x, var_id):
    """Check list of tuples.

    Function to check if the input object is None or a list of tuples.
    Other values will raise an error.

    Parameters
    ----------
    x : {List of tuple/s, None}
        The input object to check if it's a list of tuple or None, otherwise
        the function will rise an error.
    var_id : string
        The x variable id, to identify it in the error messages.

    Returns
    -------
    x : {List of tuple/s, None}
        The input parameter, returned as it was, if no errors were found.
    """
    if x is None:
        return x
    elif isinstance(x, list):
        all_tuple = [isinstance(i, tuple) for i in x]
        if all(all_tuple):
            return x
        else:
            raise ValueError(f'[ERROR] All elements of {var_id} shoud ' +
                              'be a tuple')
    else:
        raise ValueError(f'[ERROR] `{var_id}` should be a list ' +
                          'or None.')


class SpotCheck:
    """Class to spot-check algorithm performance.
    """


    def __init__(self, preprocessing = None, feature_selection = None,
                 classification = None, scoring = 'accuracy',
                 cv = None, n_jobs = 1, random_state = None):
        self.preprocessing = preprocessing
        self.feature_selection = feature_selection
        self.classification = classification
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

    @property
    def preprocessing(self):
        return self._preprocessing
    @preprocessing.setter
    def preprocessing(self, preprocessing):
        self._preprocessing = check_list_of_tuples(preprocessing,
                                                   'preprocessing')

    @property
    def feature_selection(self):
        return self._feature_selection
    @feature_selection.setter
    def feature_selection(self, feature_selection):
        self._feature_selection = check_list_of_tuples(feature_selection,
                                                       'feature_selection')

    @property
    def classification(self):
        return self._classification
    @classification.setter
    def classification(self, classification):
        self._classification = check_list_of_tuples(classification,
                                                    'classification')
