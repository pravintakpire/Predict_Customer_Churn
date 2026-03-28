# Predict Customer Churn

This is the project directory for the Kaggle competition "Predict Customer Churn". 

## Structure
- `data/`: Contains datasets
- `notebooks/`: Contains Jupyter notebooks for exploration and modeling
- `src/`: Contains source code (e.g., data processing scripts, model training scripts)
- `models/`: Contains trained model artifacts

## Environment
This project uses the Python virtual environment located at `/Users/pravintakpire/datascience/datascient`.

To activate the environment from the terminal:
```bash
source /Users/pravintakpire/datascience/datascient/bin/activate
```

The environment has also been registered as a Jupyter Kernel named `datascient`. When you open the notebooks in the `notebooks/` directory, they are pre-configured to use this kernel.

## Getting Started
1. If you haven't already, download the competition data and place it in the `data/` directory.
2. Open `notebooks/01_eda.ipynb` to begin exploratory data analysis.


kaggle competitions submit -c playground-series-s6e3 -f submission_xgboost_multiseed_100t.csv -m "XGBoost multiseed 5x10 with Hyperparameter Tuning"
