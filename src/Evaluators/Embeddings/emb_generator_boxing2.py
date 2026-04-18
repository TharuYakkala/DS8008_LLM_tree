from ..evaluate_boxing2 import (
    run_model_0,
    run_model_1,
    run_model_2,
    run_model_3,
    run_model_4
)

from pathlib import Path
import pandas as pd
import numpy as np

runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]

        
def get_emb_features_boxing2():
    data_path = Path("./data/data_sets/boxing2")
    X_path = data_path / "X.csv"
    
    X = pd.read_csv(X_path)
    
    output_emb = []
    for _, row in X.iterrows():
        row_emb = []
        for runner in runners:
            _, emb = runner(row)
            row_emb.extend(emb)
        output_emb.append(row_emb)
    return np.array(output_emb)
        






