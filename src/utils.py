import os 
import sys
import dill
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, object):
    '''
    This method saves the object to the specified file path.

    Args:
    file_path: str
        The path where the object needs to be saved
    object: object
        The object that needs to be saved
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)