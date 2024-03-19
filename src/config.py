import os

import numpy as np

INDEX_PAGE_URL = "https://www.polovniautomobili.com"
PROJECT_DIR = os.getcwd()
DTYPE_BACKEND = "numpy_nullable"
INDEX = "id"
LABEL = "price"
TEST_SIZE = 0.2
RANDOM_SEED = 2024

UNKNOWN_VALUE_BINARY = -1
UNKNOWN_VALUE_ORDINAL = np.nan
UNKNOWN_VALUE_NOMINAL = np.nan

FeaturesInfo = dict[str, list[str]]
"""Custom type alias representing a dictionary containing information about feature categories.

Structure
---------
{
    'numerical': list[str]
        List of column names for numerical features.
    'binary': list[str]
        List of column names for binary features.
    'ordinal': list[str]
        List of column names for ordinal features.
    'nominal': list[str]
        List of column names for nominal features.
    'derived_numerical': list[str]
        List of column names for derived numerical features.
    'derived_binary': list[str]
        List of column names for derived binary features.
    'derived_ordinal': list[str]
        List of column names for derived ordinal features.
    'derived_nominal': list[str]
        List of column names for derived nominal features.
    'other': list[str]
        List of other features.
    'features_to_delete': list[str]
        List of column names for features to be deleted.
}
"""
