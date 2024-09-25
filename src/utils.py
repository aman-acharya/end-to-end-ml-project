import os 
import sys
import dill
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# create a function to save the object
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

# create a function to load the object
def load_object(file_path):
    '''
    This method loads the object from the specified file path.

    Args:
    file_path: str
        The path from where the object needs to be loaded

    Returns:
    object
        The object that is loaded from the file path
    '''
    try:
        with open(file_path,'rb') as file_obj:
            object = pickle.load(file_obj)

        return object

    except Exception as e:
        raise CustomException(e, sys)

# create a function to evaluate model performance
def eval_model(X_train, X_test, y_train, y_test, models, param):
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
    param: dict
        The dictionary containing the hyperparameters for the models

    Returns:
    report: dict
        The dictionary containing the model names and their corresponding r2 scores
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = list(param.values())[i]

            gs = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_r2 = r2_score(y_train, y_train_pred)
            test_model_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)
