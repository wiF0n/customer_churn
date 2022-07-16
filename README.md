# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project contains code to train models that predict which credit card customers are most likely to churn.
- Data are from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- Two classification models trained, namely Random Forest and Logistic Regression
- EDA, model performance and feature importances are computed


## Files and data description  
.  
├── churn_notebook.ipynb  # Jupyter notebook with the analysis  
├── churn_library.py  # Core code and functions for this project (contains main() fucntion)  
├── churn_script_logging_and_tests.py # Logging and test code  
├── README.md # Project overview  
├── data  
│   └── bank_data.csv   # customer data from Kaggle  
├── images               
│   ├── eda  # Images from EDA (univariate and bivariate)
│   └── results  # ROC plot, Feat importance graphs and model performance report  
├── logs # Logs from testing
└── models  # Saved models (RF and LR)

## Running Files
To run the whole pipeline execute `churn_library.py` file.  
To run tests execute `churn_script_logging_and_tests.py` file  
