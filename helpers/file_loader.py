import pandas as pd
import numpy as np
from pathlib import Path

def load_pandas_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df