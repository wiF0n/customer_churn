"""
This module contains functionality needed to predict customer churn

Author: David Kubanda
Date: 2022-07-16
"""

# import libraries
import os

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import constants

# global settings
plt.rcParams['figure.figsize'] = (20.0, 10.0)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''

    return pd.read_csv(pth)


def perform_eda(pd_df):
    '''
    perform eda on df and save figures to images folder

    input:
            pd_df: pandas dataframe

    output:
            None
    '''
    # Print df head
    print("Head:")
    print(pd_df.head())
    print("\n")

    # Print df shape
    print("Shape:")
    print(pd_df.shape)
    print("\n")

    # Print # of null values for each column in df
    print("# of nulls:")
    print(pd_df.isnull().sum())
    print("\n")

    # Print basic statistics for each column in df
    print("Basic statistics:")
    print(pd_df.describe())
    print("\n")

    # Create histogram for target column (not transformed to int)
    sns.histplot(pd_df[constants.RAW_TARGET_COL])
    plt.savefig(os.path.join(constants.EDA_IMG_FLDR_PTH,
                             f"{constants.RAW_TARGET_COL}_hist.png"),
                bbox_inches="tight")
    plt.close()

    # Create histograms for categorical columns
    for cat_col in constants.cat_cols_lst:
        # Univariate plots
        sns.histplot(pd_df[cat_col], shrink=.8)
        plt.savefig(os.path.join(constants.EDA_IMG_FLDR_PTH,
                                 f"{cat_col}_hist.png"),
                    bbox_inches="tight")
        plt.close()

        # Bivariate plots
        sns.histplot(pd_df,
                     x=cat_col,
                     hue=constants.RAW_TARGET_COL,
                     stat="density",
                     common_norm=False,
                     multiple="dodge",
                     shrink=.8)
        plt.savefig(os.path.join(constants.EDA_IMG_FLDR_PTH,
                                 f"{cat_col}_hist_bv.png"),
                    bbox_inches="tight")
        plt.close()

    # Create histograms (density plots) for numeric columns
    for num_col in constants.quant_cols_lst:
        # Univariate
        sns.histplot(pd_df[num_col], stat='density', kde=True)
        plt.savefig(os.path.join(constants.EDA_IMG_FLDR_PTH,
                                 f"{num_col}_hist.png"),
                    bbox_inches="tight")
        plt.close()

        # Bivariate
        sns.histplot(pd_df,
                     x=num_col,
                     hue=constants.RAW_TARGET_COL,
                     common_norm=False,
                     stat='density',
                     kde=True)
        plt.savefig(os.path.join(constants.EDA_IMG_FLDR_PTH,
                                 f"{num_col}_hist_bv.png"),
                    bbox_inches="tight")
        plt.close()

    # Create correlation matrix plot
    sns.heatmap(pd_df.corr(), annot=False, cmap='viridis', linewidths=2)
    plt.savefig(os.path.join(constants.EDA_IMG_FLDR_PTH, "corr_mtx.png"),
                bbox_inches="tight")
    plt.close()


def encoder_helper(pd_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            pd_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
            pd_df: pandas dataframe with new columns for
    '''
    # Loop through the categorical columns
    for cat_col in category_lst:
        cat_col_lst = []
        # Compute proportion of churn for each level of the variable
        cat_col_grps = pd_df.groupby(cat_col).mean()[response]

        # Crete list of the proportion of churn based on the level
        for val in pd_df[cat_col]:
            cat_col_lst.append(cat_col_grps.loc[val])

        # Create new numeric column based on previous list
        pd_df[f"{cat_col}_{response}"] = cat_col_lst

    return pd_df


def perform_feature_engineering(pd_df, response):
    '''
    input:
              pd_df: pandas dataframe
              response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create target (response)
    pd_df[response] = pd_df[constants.RAW_TARGET_COL].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Encode categorical variables
    pd_df = encoder_helper(pd_df, constants.cat_cols_lst, response)

    # Define X and y
    x_df = pd.DataFrame()
    x_df[constants.modeling_cols_lst] = pd_df[constants.modeling_cols_lst]
    y_df = pd_df[response]

    # Perform train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_df,
        y_df,
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_STATE)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Loop through model architectures
    for model_arch in constants.model_arch_lst:
        if model_arch == "rf":
            model_nm = "Random Forest"
            y_test_preds = y_test_preds_rf
            y_train_preds = y_train_preds_rf

        elif model_arch == "lr":
            model_nm = "Logistic Regression"
            y_test_preds = y_test_preds_lr
            y_train_preds = y_train_preds_lr

        # Set size of the fig
        plt.rc('figure', figsize=(5, 5))

        # Create and save the report
        plt.text(0.01,
                 1.25,
                 str(f'{model_nm} Train'),
                 {'fontsize': constants.CLS_RPRT_FNT_SIZE},
                 fontproperties=constants.CLS_RPRT_FNT)
        plt.text(0.01,
                 0.05,
                 str(classification_report(y_train, y_train_preds)),
                 {'fontsize': constants.CLS_RPRT_FNT_SIZE},
                 fontproperties=constants.CLS_RPRT_FNT)
        plt.text(0.01,
                 0.6,
                 str(f'{model_nm} Test'), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01,
                 0.7,
                 str(classification_report(y_test, y_test_preds)),
                 {'fontsize': constants.CLS_RPRT_FNT_SIZE},
                 fontproperties=constants.CLS_RPRT_FNT)
        plt.axis('off')
        plt.savefig(os.path.join(constants.RSLT_IMG_FLDR_PTH,
                                 f"{model_arch}_cls_rprt.png"),
                    bbox_inches="tight")
        plt.close()


def roc_plot(rfc_model, lr_model, x_test, y_test):
    '''
    Produces ROC plot for test split and stores it as image in images folder
    input:
            model: fitted model
            x_df:  pandas datafame with features
            y_df: pandas datafame with target

    output:
             None
    '''
    axis = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(lr_model, x_test, y_test, ax=axis, alpha=0.8)
    plt.title("ROC curve for test split")
    plt.savefig(constants.ROC_IMG_PTH, bbox_inches="tight")
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    plt.savefig(output_pth, bbox_inches="tight")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # Define RF and LR models
    rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    lrc = LogisticRegression(solver=constants.LR_SOLVER,
                             max_iter=constants.LR_MAX_ITER)

    # Fit RF
    cv_rfc = GridSearchCV(estimator=rfc,
                          param_grid=constants.rf_param_grid_dict,
                          cv=constants.RF_NUM_CV_FOLDS)

    cv_rfc.fit(x_train, y_train)

    # Fit LF
    lrc.fit(x_train, y_train)

    # Save the models
    joblib.dump(cv_rfc.best_estimator_, constants.RF_MODEL_PTH)
    joblib.dump(lrc, constants.LR_MODEL_PTH)

    # Load the models

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Make prediction (on both test and train df)
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    # Create modeling results report
    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf)

    # Create ROC plots
    roc_plot(rfc_model, lr_model, x_test, y_test)

    # Create feature importance plots
    feature_importance_plot(rfc_model, x_test, constants.FI_IMG_PTH)


def main():
    """
    Executes whole churn prediction pipeline
    """

    # Load data
    churn_df = import_data(constants.DATA_PTH)

    # Perform EDA
    perform_eda(churn_df)

    # Perform feature eng
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        churn_df, constants.TARGET_COL)

    # Train, save and test models
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
