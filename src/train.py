# Testing the PR trigger for AI-300
import argparse
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow

mlflow_autolog()
def main():
    # 1. Handle Arguments (Matches the 'inputs' in your job.yml)
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str)
    args, unknown = parser.parse_known_args()

    # 2. Load Data
    print(f"Loading data from: {args.training_data}")
    df = pd.read_csv(args.training_data)

    # 3. Simple Training Logic
    X = df.drop('Diabetic', axis=1).values
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    print("Training model...")
    model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)

    # 4. Log Metrics (This is what shows up in Azure ML Studio)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    
    # Optional: Save model for registration later
    os.makedirs('outputs', exist_ok=True)
    # joblib.dump(value=model, filename='outputs/diabetes_model.pkl')
    print("AI-300 Experiment: running version 2.0 with GitHub Environments")

if __name__ == "__main__":
    main()
