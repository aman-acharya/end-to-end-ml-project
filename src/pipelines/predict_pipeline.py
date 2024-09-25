import os 
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import StandardScaler

class PredictionPipeline:
    '''
    This class is used to predict the target variable based on the input features.
    '''
    def __init__(self) -> None:
        pass

    def predict(self, features):
        '''
        This method predicts the target variable based on the input features.

        Args:
        features: pd.DataFrame
            The input features based on which the target variable needs to be predicted

        Returns:
        np.ndarray
            The predicted target variable
        '''
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    '''
    This class is used to create a custom data object.
    '''
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                math_score: int,
                reading_score: int):
        
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.math_score = math_score

        self.reading_score = reading_score

    def get_data_as_dataframe(self):
        '''
        This method returns the custom data as a pandas DataFrame.

        Returns:
        pd.DataFrame
            The custom data as a pandas DataFrame
        '''
        try:
            custom_data_dict = {
                    "gender" : [self.gender],
                    "race_ethnicity" : [self.race_ethnicity],
                    "parental_level_of_education" : [self.parental_level_of_education],
                    "lunch" : [self.lunch],
                    "test_preparation_course" : [self.test_preparation_course],
                    "math_score" : [self.math_score],
                    "reading_score" : [self.reading_score]
            }

            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
