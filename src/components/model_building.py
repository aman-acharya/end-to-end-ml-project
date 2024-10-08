import os
import sys
from urllib.parse import urlparse
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import mlflow
from sklearn.metrics import r2_score
from src.utils import eval_model, save_object, save_model_metrics

@dataclass
class ModelBuildingConfig:
    '''
    This class holds the configuration for the model building component.
    '''
    model_file_path:str = os.path.join('artifacts', "model.pkl")
    train_metrics:str = os.path.join('artifacts', "train_metrics.json")
    test_metrics:str = os.path.join('artifacts', "test_metrics.json")


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
                'XGBRegressor': XGBRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor()
            }

            params = {
                'Linear Regression' : {},

                'Ridge': {'alpha': [0.1, 1, 3, 5, 7, 10]},

                'Lasso': {'alpha': [0.1, 1, 3, 5, 7, 10]},

                'Random Forest Regressor' : {
                    'n_estimators': [8,16,32,64,128,256], 
                    'max_depth': [3, 5, 7, 9]
                    },

                'K-Neighbors Regressor' : {
                    'n_neighbors': [3, 5, 7, 9]
                    },

                'XGBRegressor': {
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3]
                    },

                'AdaBoost Regressor': {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3]
                    },

                'Gradient Boosting Regressor': {
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3]
                    }
            }

            logging.info('Successfully built the models')

            training_metrics, test_metrics = eval_model(X_train, X_test, y_train, y_test, models, params)
            
            logging.info(f"Model metrics have been saved successfully at {self.model_building_config.train_metrics} and {self.model_building_config.test_metrics}")

            # Select the best model based on test accuracy
            best_model_score = max(test_metrics.values(), key=lambda x: x['r2_score'])['r2_score']
            best_model_name = max(test_metrics, key=lambda x: test_metrics[x]['r2_score'])

            best_model = models[best_model_name]

            logging.info(f"The best model is {best_model_name} with an accuracy score of {best_model_score}")

            logging.info("Tracking the best model using MLflow")
            actual_model = ""
            for model in models:
                if model == best_model_name:
                    actual_model = actual_model + model

            mlflow.set_registry_uri("https://dagshub.com/ayushach007/end-to-end-ml-project.mlflow")
            tracking_uri_type = urlparse(mlflow.get_registry_uri()).scheme

            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("best parameters", test_metrics[model_name]['best_params'])
                    mlflow.log_metric("training_r2_score", training_metrics[model_name]['r2_score'])
                    mlflow.log_metric("training_mae", training_metrics[model_name]['mae'])
                    mlflow.log_metric("training_mse", training_metrics[model_name]['mse'])
                    mlflow.log_metric("training_rmse", training_metrics[model_name]['rmse'])
                    mlflow.log_metric("test_r2_score", test_metrics[model_name]['r2_score'])
                    mlflow.log_metric("test_mae", test_metrics[model_name]['mae'])
                    mlflow.log_metric("test_mse", test_metrics[model_name]['mse'])
                    mlflow.log_metric("test_rmse", test_metrics[model_name]['rmse'])
                    mlflow.sklearn.log_model(model, "model")

            
            if tracking_uri_type != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.7:
                logging.warning('The best model has an r2 score of less than 0.5. Please consider retraining the model.')

            logging.info(f'The best model is {best_model_name} with an r2 score of {best_model_score}')

            save_object(
                file_path = self.model_building_config.model_file_path,
                object = models[best_model_name]
            )
            logging.info(f"Model has been saved successfully at {self.model_building_config.model_file_path}")

            # Save model metrics
            save_model_metrics(
                report = training_metrics,
                path = self.model_building_config.train_metrics
            )

            save_model_metrics(
                report = test_metrics,
                path = self.model_building_config.test_metrics
            )

            logging.info(f"Model metrics have been saved successfully at {self.model_building_config.train_metrics} and {self.model_building_config.test_metrics}")

            predicted=best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e, sys)