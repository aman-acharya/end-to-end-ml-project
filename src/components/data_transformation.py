import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    This class holds the configuration for the data transformation component.
    '''
    preprocessor_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprojessor_obj(self):
        '''
        This method creates a preprocessor object that can be used to transform the data.
        
        Returns:
        preprocessor: ColumnTransformer
            The preprocessor object that can be used to transform the data.
        '''

        try:
            logging.info('Starting to create preprocessor object')
            numerical_features = ['math_score', 'reading_score']

            categorical_features = [
                    'gender',
                    'race_ethnicity',	
                    'parental_level_of_education',
                    'lunch',
                    'test_preparation_course'
            ]

            logging.info('Starting to create numerical pipeline')
            num_pipeline = Pipeline(
                    steps = [
                        ('imputer', SimpleImputer(strategy = 'most_frequent')),
                        ('scaler', StandardScaler())
                    ]
            )
            logging.info('Successfully created numerical pipeline')

            logging.info('Starting to create categorical pipeline')
            cat_pipeline = Pipeline(
                    steps = [
                        ('imputer', SimpleImputer(strategy = 'most_frequent')),
                        ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
                        ('scaler', StandardScaler())
                    ]
            )
            logging.info('Successfully created categorical pipeline')

            logging.info('Starting to merge numerical and categorical pipelines for data preprocessing.')
            preprocessor = ColumnTransformer(
                    [
                        ('num_pipeline', num_pipeline, numerical_features),
                        ('cat_pipeline', cat_pipeline, categorical_features)
                    ]
            )
            logging.info('Successfully completed merging numerical and categorical pipelines.')
            logging.info('Successfully created preprocessor object.')
        
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This method reads the train and test data and transforms it using the preprocessor object.

        Args:
        train_path: str
            The path of the train data
        test_path: str
            The path of the test data

        Returns:
        train_arr: np.ndarray
            The transformed training data
        test_arr: np.ndarray
            The transformed test data
        preprocessor_file_path: str
            The path where the preprocessor object is saved
        '''

        try:
            logging.info('Initiating data transformation process.')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Successfully read train and test data')

            logging.info('Obtaining the preprocessor object for data transformation.')

            preprocessor_object = self.get_preprojessor_obj()

            target_feature = "writing_score"

            train_df_input_features = train_df.drop(columns=[target_feature], axis=1)
            test_df_input_features = test_df.drop(columns=[target_feature], axis=1)

            train_df_target = train_df[target_feature]
            test_df_target = test_df[target_feature]

            logging.info('Fitting the preprocessor on the training and test datasets to prepare for model training and evaluation.')

            train_arr_input_feature = preprocessor_object.fit_transform(train_df_input_features)
            test_arr_input_feature = preprocessor_object.transform(test_df_input_features)

            train_arr = np.c_[
                    train_arr_input_feature, np.array(train_df_target)
                    ]
            
            test_arr = np.c_[
                        test_arr_input_feature, np.array(test_df_target)
                    ]
            
            logging.info('Successfully fitted the preprocessor on the training and test datasets')

            logging.info('Saving the preprocessor object in a pickle file')
            save_object(
                file_path= self.data_transformation_config.preprocessor_file_path,
                object=preprocessor_object
            )

            logging.info('Successfully saved the preprocessor object to the specified file path.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


