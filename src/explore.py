import pandas as pd
import sys

def check_data():
    try:
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')
        sub = pd.read_csv('data/sample_submission.csv')
        
        print("--- TRAIN SET ---")
        print(f"Shape: {train.shape}")
        print("\nColumns:")
        print(train.dtypes)
        print("\nMissing values:")
        print(train.isnull().sum()[train.isnull().sum() > 0])
        print("\nHead:")
        print(train.head(3))
        
        print("\n\n--- TEST SET ---")
        print(f"Shape: {test.shape}")
        
        print("\n\n--- SUBMISSION SET ---")
        print(f"Shape: {sub.shape}")
        print(sub.head(2))
        
    except Exception as e:
        print(f"Error reading data: {e}")

if __name__ == "__main__":
    check_data()
