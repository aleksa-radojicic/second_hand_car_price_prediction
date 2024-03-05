import os
from typing import Dict, List

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

FeaturesInfo = Dict[str, List[str]]
"""Custom type alias representing a dictionary containing information about feature categories.

Structure
---------
{
    'numerical': List[str]
        List of column names for numerical features.
    'binary': List[str]
        List of column names for binary features.
    'ordinal': List[str]
        List of column names for ordinal features.
    'nominal': List[str]
        List of column names for nominal features.
    'derived_numerical': List[str]
        List of column names for derived numerical features.
    'derived_binary': List[str]
        List of column names for derived binary features.
    'derived_ordinal': List[str]
        List of column names for derived ordinal features.
    'derived_nominal': List[str]
        List of column names for derived nominal features.
    'other': List[str]
        List of other features.
    'features_to_delete': List[str]
        List of column names for features to be deleted.
}
"""