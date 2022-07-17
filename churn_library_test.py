"""
Loggign and testing for churn prediciton code

Author: David Kubanda
Date: 2022-07-17
"""

# import libraries
import os
import logging
import joblib

import pytest

import constants
import churn_library as cls

logging.basicConfig(filename=constants.LOG_PTH,
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name="raw_df")
def raw_df_():
    """
    Raw df fixture - returns initial raw df
    """
    try:
        raw_df = cls.import_data(constants.DATA_PTH)
        logging.info("Importing raw data (as a fixture): SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Importing raw data (as a fixture): The file wasn't found")
        raise err

    return raw_df


@pytest.fixture(name="encoded_df")
def encoded_df_(raw_df):
    """
    Encoded df fixture - returns df with encoded categorical columns and new target variable
    """
    try:
        encoded_df = cls.encoder_helper(raw_df, constants.cat_cols_lst,
                                        constants.TARGET_COL)
        logging.info("Encoding raw data (as a fixture): SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoding raw data (as a fixture): Categorical column not found in data"
        )
        raise err

    return encoded_df


@pytest.fixture(name="modeling_data")
def modeling_data_(encoded_df):
    """
    Modeling data fixture - returns X/y train/test split of data
    """
    try:
        x_train, x_test, y_train, y_test = cls.perform_feat_eng(
            encoded_df, constants.TARGET_COL)
        logging.info("Creating modeling data: SUCCESS")
    except Exception as err:
        logging.error("Creating modeling data: FAILED")
        raise err

    return x_train, x_test, y_train, y_test


def test_import(raw_df):
    '''
	test data import - test if that the dataframe has rows and cols
    (testing whether the file exists has been already done in raw_df fixture)
	'''

    try:
        assert raw_df.shape[0] > 0
        assert raw_df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_perform_eda(raw_df):
    '''
	test perform eda function - test if the function creates images
	'''

    # Check whether image folder exists
    try:
        assert os.path.isdir(constants.EDA_IMG_FLDR_PTH)
        logging.info("Testing perform_eda: Folder for images exists")
    except AssertionError as err:
        logging.error("Testing perform_eda: Folder for images does NOT exist")
        raise err

    # Remove all contents from the image folder
    for file in os.listdir(constants.EDA_IMG_FLDR_PTH):
        os.remove(os.path.join(constants.EDA_IMG_FLDR_PTH, file))

    # Perform EDA
    cls.perform_eda(raw_df)

    # Check whether files have been created
    try:
        assert len(os.listdir(constants.EDA_IMG_FLDR_PTH)) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: No images were created")
        raise err


def test_encoder_helper(encoded_df, raw_df):
    '''
	test encoder helper - check if the encoded and raw df have the same amount of rows
	'''
    try:
        assert encoded_df.shape[0] == raw_df.shape[0]
    except AssertionError as err:
        logging.error("Testing encoder_helper: \
                Encoded df must have the same amount of rows as raw_df")
        raise err


def test_perform_feat_eng(modeling_data):
    '''
	test perform_feat_eng - check if the # rows between X and y
        as well as # cols between X train and X test is equal
	'''
    x_train, x_test, y_train, y_test = modeling_data

    # Number of rows for X and y must be the same
    try:
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        logging.info("Testing perform_feat_eng # rows: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feat_eng # rows: FAILED")
        raise err

    # Number of cols for X train and X test must be the same
    try:
        assert x_train.shape[1] == x_test.shape[1]
        logging.info("Testing perform_feat_eng # cols: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feat_eng # cols: FAILED")
        raise err


def test_train_models(modeling_data):
    '''
	test train_models - check if the models were created
	'''
    # Remove model files
    for file in os.listdir(constants.MODEL_FLDR_PTH):
        os.remove(os.path.join(constants.MODEL_FLDR_PTH, file))

    # Train models
    cls.train_models(*modeling_data)

    # Test if the model picke files has been created
    try:
        joblib.load(constants.RF_MODEL_PTH)
        joblib.load(constants.LR_MODEL_PTH)
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models: Model picke file has NOT been found")
        raise err


def test_cls_rprt(modeling_data):
    '''
	test cls_rprt - check if the report was created
	'''
    # Get data
    x_train, x_test, y_train, y_test = modeling_data

    # Load models
    rfc_model = joblib.load(constants.RF_MODEL_PTH)
    lr_model = joblib.load(constants.LR_MODEL_PTH)

    # Make prediction (on both test and train df)
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    y_data = y_train, y_test
    y_pred_data = y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf

    # Remove all contents of results folder
    for file in os.listdir(constants.RSLT_IMG_FLDR_PTH):
        os.remove(os.path.join(constants.RSLT_IMG_FLDR_PTH, file))

    # Make reports
    cls.cls_rprt(y_data, y_pred_data)

    # Test if the reports have been created
    try:
        assert len(os.listdir(constants.RSLT_IMG_FLDR_PTH)) > 0
        logging.info("Testing cls_rprt: SUCCESS")
    except AssertionError as err:
        logging.error("Testing cls_rprt: Reports have not been created")
        raise err


def test_roc_plot(modeling_data):
    '''
    test roc_plot - check if the ROC plot was created
    '''
    # Get data
    _, x_test, _, y_test = modeling_data

    # Load models
    rfc_model = joblib.load(constants.RF_MODEL_PTH)
    lr_model = joblib.load(constants.LR_MODEL_PTH)

    # Create ROC plot
    cls.roc_plot(rfc_model, lr_model, x_test, y_test)

    # Test if the plot has been created
    try:
        assert os.path.exists(constants.ROC_IMG_PTH)
        logging.info("Testing roc_plot: SUCCESS")
    except AssertionError as err:
        logging.error("Testing roc_plot: Plot has not been created")
        raise err


def test_feat_imp_plot(modeling_data):
    '''
    test feat_imp_plot - check if the feature importance plot was created
    '''
    # Get data
    _, x_test, _, _ = modeling_data

    # Load RF model
    rfc_model = joblib.load(constants.RF_MODEL_PTH)

    # Create FI plot
    cls.feat_imp_plot(rfc_model, x_test, constants.FI_IMG_PTH)

    # Test if the plot has been created
    try:
        assert os.path.exists(constants.FI_IMG_PTH)
        logging.info("Testing roc_plot: SUCCESS")
    except AssertionError as err:
        logging.error("Testing roc_plot: Plot has not been created")
        raise err


if __name__ == "__main__":
    # Run the tests
    pytest.main()
