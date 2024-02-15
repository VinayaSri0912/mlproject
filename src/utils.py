import dill
import os
import sys
from src.exception import CustomException

def save_obj(obj, path):
    try :
        with open(path, "wb") as f:
            dill.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e, sys)
        