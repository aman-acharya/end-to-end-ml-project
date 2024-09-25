import os
import sys
import pandas as pd
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_building import ModelBuilding, ModelBuildingConfig

class TrainPipeline:
    '''
    This class is used to train the model based on the input data.
    '''
    def __init__(self) -> None:
        pass

    def train(self):
        '''
        This method trains the model based on the input data.

        Args:
        data_path: str
            The path where the data is stored
        config: dict
            The configuration parameters required for training the model
        '''
        try:
            obj = DataIngestion()
            train_data, test_data = obj.initiate_data_ingestion()

            data_transformation_obj = DataTransformation()
            train_arr, test_arr, _ = data_transformation_obj.initiate_data_transformation(train_data, test_data)

            model_building_obj = ModelBuilding()
            model_building_obj.initiate_model_building(train_arr, test_arr) 

        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.train()