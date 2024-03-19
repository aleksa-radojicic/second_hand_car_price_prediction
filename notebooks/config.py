import os

INDEX_PAGE_URL = "https://www.polovniautomobili.com"
PROJECT_DIR = os.getcwd()
DTYPE_BACKEND = "numpy_nullable"
INDEX = "id"
LABEL = "price"
TEST_SIZE = 0.2
RANDOM_SEED = 2024

FeaturesInfo = dict[str, list[str]]
"""Custom type alias representing a dictionary containing information about feature categories.

Structure
---------
{
    'numerical': list[str]
        list of column names for numerical features.
    'binary': list[str]
        list of column names for binary features.
    'ordinal': list[str]
        list of column names for ordinal features.
    'nominal': list[str]
        list of column names for nominal features.
    'derived_numerical': list[str]
        list of column names for derived numerical features.
    'derived_binary': list[str]
        list of column names for derived binary features.
    'derived_ordinal': list[str]
        list of column names for derived ordinal features.
    'derived_nominal': list[str]
        list of column names for derived nominal features.
    'other': list[str]
        list of other features.
    'features_to_delete': list[str]
        list of column names for features to be deleted.
}
"""