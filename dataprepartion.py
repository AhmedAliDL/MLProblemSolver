import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPrepare:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def remove_col(self, data, cols):
        """this function remove specific column from dataframe"""
        try:
            return data.drop(cols, axis=1, inplace=True)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def fill_mean(self, data, cols):
        """this function fill null values of column with mean"""
        try:
            for col in cols:
                data[col].fillna(value=data[col].mean(), inplace=True)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def fill_mode(self, data, cols):
        """this function fill null values of column with mode"""
        try:
            for col in cols:
                data[col].fillna(value=data[col].mode(), inplace=True)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def fill_value(self, data, cols, value):
        """this function fill null values of column with value"""
        try:
            for col in cols:
                data[col].fillna(value=value, inplace=True)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def fill_forward(self, data, cols):
        """this function fill null values of column with forward value"""
        try:
            for col in cols:
                data[col].ffill(inplace=True)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def fill_backward(self, data, cols):
        """this function fill null values of column with backward value"""
        try:
            for col in cols:
                data[col].bfill(inplace=True)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def get_dummy(self, data, cols):
        """this function convert categorical values to numerical values
        by just using one-hot-encoding method"""
        try:
            return pd.get_dummies(data, columns=cols)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def scaler(self, data, col):
        """this function normalize column values to be between 0 and 1"""
        try:
            return MinMaxScaler().fit_transform(data[col])
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def select_data(self, data, output, train_size):
        """this function splitting data by size of train
        using train_test_split function"""
        try:
            return train_test_split(data[data.columns[data.columns != output]],
                                    data[output], train_size=train_size)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None
