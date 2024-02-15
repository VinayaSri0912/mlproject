import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
import sys
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass


class DataTransConfig:
    processor_path:str=os.path.join("artifacts","processor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transform_config= DataTransConfig()
    def get_data_transform_obj(self,train_df ):
        
        try :
        
            numerical_columns = list(train_df.select_dtypes(include=['number']).columns)
            categorical_columns = list(train_df.select_dtypes(include=['object']).columns)
            
            # logging.info(numerical_columns, categorical_columns)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]    
            )
            
            cat_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean= False))
                    ]
            )
            
            logging.info("numerica,categorical done")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ],
                remainder= "drop"
                
            )
            return preprocessor
        
    
        except Exception as e :
            raise CustomException(e, sys)
        
    
    
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(train_df.head())
            
            
            target_column_name = "math_score"
            
            input_features_train = train_df.drop(target_column_name,axis=1)
            target_features_train = train_df[target_column_name]
            preprocessing_obj = self.get_data_transform_obj(input_features_train)

            
            numerical_columns = list(input_features_train.select_dtypes(include=['number']).columns)
            
            logging.info(numerical_columns)
            
            
            
            input_features_test = test_df.drop(columns=[target_column_name],axis=1)
            target_features_test = test_df[target_column_name]
            
            # logging.info(target_features_train, target_features_test)
            
            logging.info(input_features_train)
            input_train = preprocessing_obj.fit_transform(input_features_train)
            input_test = preprocessing_obj.transform(input_features_test)
            
            train_arr = np.c_[input_train, np.array(target_features_train)]
            test_arr = np.c_[input_test, np.array(target_features_test)]
            
            save_object(
                preprocessing_obj,
                self.data_transform_config.processor_path
            )
            
            return (
                preprocessing_obj,
                train_arr,
                test_arr
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            

            
        
    