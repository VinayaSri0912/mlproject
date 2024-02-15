import os
import sys



from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path :str = os.path.join("artifacts", "train_data.csv")
    test_path :str = os.path.join("artifacts", "test_data.csv")
    raw_path :str = os.path.join("artifacts", "raw_data.csv")
    
class DataIngestion :
    def __init__(self):
        self.dataConfig = DataIngestionConfig()
        
    def start_dataIngestion(self):
        
        try :
            df = pd.read_csv("notebooks\data.csv")
            
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=5)
            
            os.makedirs(os.path.dirname(self.dataConfig.raw_path), exist_ok=True)
            df.to_csv(self.dataConfig.raw_path, index=False)
            
            df.to_csv(self.dataConfig.train_path, index=False)
            df.to_csv(self.dataConfig.test_path, index=False)
            
            return (
                self.dataConfig.train_path,
                self.dataConfig.test_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    train_path, test_path = DataIngestion().start_dataIngestion()
    data_transformation = DataTransformation()
    _, train_arr, test_arr= data_transformation.initiate_data_transform(train_path, test_path)
    model = ModelTrainer()
    print(model.initiate_model_trainer(train_arr, test_arr))
    
            
        
        
        






