import os 
import sys
import dill
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
def eval_model(X_train, X_test, y_train, y_test, models, param) -> dict:
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
        train_metrics = {}
        test_metrics = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
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

            train_model_mae = mean_absolute_error(y_train, y_train_pred)
            test_model_mae = mean_absolute_error(y_test, y_test_pred)

            train_model_mse = mean_squared_error(y_train, y_train_pred)
            test_model_mse = mean_squared_error(y_test, y_test_pred)

            train_model_rmse = np.sqrt(train_model_mse)
            test_model_rmse = np.sqrt(test_model_mse)

            train_metrics[model_name] = {
                'r2_score': train_model_r2,
                'mae': train_model_mae,
                'mse': train_model_mse,
                'rmse': train_model_rmse,
                'best_params' : gs.best_params_
            }

            test_metrics[model_name] = {
                'r2_score': test_model_r2,
                'mae': test_model_mae,
                'mse': test_model_mse,
                'rmse': test_model_rmse,
                'best_params' : gs.best_params_
            }


        return train_metrics, test_metrics

    except Exception as e:
        raise CustomException(e, sys)

# create a function to save the model metrics
def save_model_metrics(report, path):
    '''
    This function saves the model metrics to the specified path in json format

    Args:
        report: dict
            The dictionary containing the model names and their corresponding accuracy scores
        path: str
            Path to save the model metrics

    Raises:
        CustomException: If there is an error saving the model metrics
    '''
    try:
        # Convert any non-serializable objects to serializable format
        serializable_report = {}
        for model_name, metrics in report.items():
            serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
            serializable_metrics["model"] = model_name
            serializable_report[model_name] = serializable_metrics

        logging.info(f"Saving model metrics to path {path}")
        with open(path, 'w') as f:
            json.dump(serializable_report, f, indent=4, sort_keys=True)
        logging.info(f"Model metrics saved to path {path}")
    except Exception as e:
        raise CustomException(e, sys)