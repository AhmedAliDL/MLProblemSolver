import logging
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Model:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def rfc(self, X_train, y_train, n_est, r_state=42):
        """this function just using Random forest classification model
        for training and adjust some parameters of this model"""
        try:
            model = RandomForestClassifier(n_estimators=n_est,
                                           random_state=r_state)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def xgbc(self, X_train, y_train, n_est, r_state=42):
        """this function just using Extreme Gradient Boosting classification
         model for training and adjust some parameters of this model"""
        try:
            model = XGBClassifier(n_estimators=n_est,
                                  random_state=r_state)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def lgbmc(self, X_train, y_train, n_est, r_state=42):
        """this function just using light gradient boosting machine
         classification model for training and adjust some parameters
         of this model"""
        try:
            model = LGBMClassifier(n_estimators=n_est,
                                   random_state=r_state)
            model.fit(self, X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def rfr(self, X_train, y_train, n_est, r_state=42):
        """this function just using Random forest regression model
                for training and adjust some parameters of this model"""
        try:
            model = RandomForestRegressor(n_estimators=n_est,
                                          random_state=r_state)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def xgbr(self, X_train, y_train, n_est, r_state=42):
        """this function just using Extreme Gradient Boosting regression
                 model for training and adjust some parameters of this model"""
        try:
            model = XGBRegressor(n_estimators=n_est,
                                 random_state=r_state)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def lgbmr(self, X_train, y_train, n_est, r_state=42):
        """this function just using light gradient boosting machine
                 regression model for training and adjust some parameters
                 of this model"""
        try:
            model = LGBMRegressor(n_estimators=n_est,
                                  random_state=r_state)
            model.fit(self, X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None
