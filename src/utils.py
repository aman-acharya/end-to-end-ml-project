import os 
import sys
import dill
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

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
    
# create a function to evaluate model performance
def eval_model(X_train, X_test, y_train, y_test, models):
    '''
    This method evaluates the performance of the models on the test data.
    
    Args:
    X_train: np.ndarray
        The input features of the training data
    X_test: np.ndarray
        The input features of the test data
    y_train: np.ndarray
        The target feature of the training data
    y_test: np.ndarray
        The target feature of the test data
    models: dict
        The dictionary containing the model objects

    Returns:
    report: dict
        The dictionary containing the model names and their corresponding r2 scores
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_r2 = r2_score(y_train, y_train_pred)
            test_model_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)
