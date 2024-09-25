import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils import eval_model, save_object

@dataclass
class ModelBuildingConfig:
    '''
    This class holds the configuration for the model building component.
    '''
    model_file_path:str = os.path.join('artifacts', "model.pkl")


class ModelBuilding:
    '''
    This class is responsible for building the models and evaluating their performance.
    '''
    def __init__(self):
        self.model_building_config = ModelBuildingConfig()

    def initiate_model_building(self, train_arr, test_arr):
        '''
        This method builds the models and evaluates their performance.

        Args:
        train_arr: np.ndarray
            The training data
        test_arr: np.ndarray
            The test data
        
        Returns:
        best_model_name: str
            The name of the best model
        best_model_score: float
            The r2 score of the best model
        '''

        try:
            logging.info('Splitting the train and test data')

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info('Successfully split the data for model training')

            logging.info('Starting the model building process')

            models = {
                'Linear Regression' : LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Random Forest Regressor' : RandomForestRegressor(),
                'K-Neighbors Regressor' : KNeighborsRegressor(),
                'SVM': SVR(),
                'XGBRegressor': XGBRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            model_report:dict = eval_model(X_train = X_train, X_test = X_test,
                                            y_train = y_train, y_test = y_test, 
                                            models = models)

            logging.info('Successfully built the models')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.7:
                logging.warning('The best model has an r2 score of less than 0.5. Please consider retraining the model.')

            logging.info(f'The best model is {best_model_name} with an r2 score of {best_model_score}')

            save_object(
                file_path = self.model_building_config.model_file_path,
                object = models[best_model_name]
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)