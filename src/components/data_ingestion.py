import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        This method reads the data from the source and splits it into train and test data.

        Returns:
        train_data_path: str
            The path of the train data
        test_data_path: str
            The path of the test data
        '''
        logging.info('Entered the data ingestion method or component')

        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Data read successfully')

            # Ensure the directory for the train data path exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Ensure the directory for the test data path exists
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # Ensure the directory for the raw data path exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train-Test split initiated')

            # Split the data into train and test
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            logging.info('Train-Test split completed')

            # Save the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()