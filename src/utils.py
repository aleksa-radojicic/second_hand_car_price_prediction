
from src.config import FeaturesInfo


def initialize_features_info() -> FeaturesInfo:
    features_info: FeaturesInfo = {
        'numerical': [],
        'binary': [],
        'ordinal': [],
        'nominal': [],
        'derived_numerical': [],
        'derived_binary': [],
        'derived_ordinal': [],
        'derived_nominal': [],
        'other': [],
        'features_to_delete': []
    }
    return features_info
    