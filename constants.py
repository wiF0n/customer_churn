"""
This module contains all the constants used in churn prediction

Author: David Kubanda
Date: 2022-07-16
"""

# Data related constants
DATA_PTH = r"./data/bank_data.csv"

cat_cols_lst = [
    'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
    'Card_Category'
]

quant_cols_lst = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

modeling_cols_lst = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]

RAW_TARGET_COL = "Attrition_Flag"
TARGET_COL = "Churn"

# EDA related constants

EDA_IMG_FLDR_PTH = "./images/eda/"

# Modeling related constants

model_arch_lst = ["rf", "lr"]

TEST_SIZE = 0.3
RANDOM_STATE = 42

LR_SOLVER = "lbfgs"
LR_MAX_ITER = 3000

rf_param_grid_dict = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

RF_NUM_CV_FOLDS = 5

RSLT_IMG_FLDR_PTH = "./images/results/"
MODEL_FLDR_PTH = "./models/"
RF_MODEL_PTH = './models/rfc_model.pkl'
LR_MODEL_PTH = './models/logistic_model.pkl'
ROC_IMG_PTH = "./images/results/test_au_roc.jpg"
FI_IMG_PTH = "./images/results/feat_imp.jpg"

CLS_RPRT_FNT = "monospace"
CLS_RPRT_FNT_SIZE = 10

# Logging and testing constants

LOG_PTH = './logs/churn_library.log'
