import pandas as pd
import numpy as np
from helpers.file_loader import load_pandas_data



def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    