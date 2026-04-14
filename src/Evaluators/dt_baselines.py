from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def get_baseline_dt(dataset_name: str):
    data_path = "./data/data_sets"
    # Train and fit a dt model to the dataset to get a baseline prediction
  
    path = Path(data_path) / dataset_name
    x_path = path / "X.csv"
    y_path = path / "y.csv"
        
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("--|| Baseline DT Model ||--")
    print(classification_report(y_test, pred))
    