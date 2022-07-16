"""
This module contains all the constants used in churn prediction

Author: David Kubanda
Date: 2022-07-16
"""

DATA_PTH = r"./data/bank_data.csv"

cat_columns_list = [
    'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
    'Card_Category'
]

quant_columns_list = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]
