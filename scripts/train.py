import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import os

def train_model():
    print("Loading clean dataset...")
    df = pd.read_csv("data/cleaned_monthly_ev.csv")
    
    # 1. Feature Engineering: Create a numerical time index for regression
    df['Time_Index'] = np.arange(len(df))
    X = df[['Time_Index']]
    y = df['Registrations']
    
    # 2. Train the Model
    print("Training baseline Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Serialize and Save the Model
    os.makedirs("api", exist_ok=True)
    joblib.dump(model, "api/ev_model.joblib")
    
    print(f"EV-2 Success: Model trained and saved to api/ev_model.joblib")
    print(f"Model R^2 Score: {model.score(X, y):.4f}")

if __name__ == "__main__":
    train_model()