
import os
import pickle
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
    

def pickle_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e


def json_object(file_path, obj):
    import json

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file_obj:
            json.dump(obj, file_obj, indent=1)

    except Exception as e:
        raise e