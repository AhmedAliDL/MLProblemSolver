import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use(r"https://raw.githubusercontent.com/dhaitz/matplotlib-stylesheets/master/pitayasmoothie-dark.mplstyle")


class EDA:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def data_dist(self, data, col, fig_size):
        """this function showing data distribution of object and numerical
        data"""
        try:
            if data[col].dtype == object:
                data_count = data[col].value_counts()
                plt.figure(figsize=fig_size)
                sns.barplot(x=data_count.values,
                            y=data_count.keys()
                            , width=0.3)
                st.pyplot(plt)
            else:
                plt.figure(figsize=fig_size)
                ax = sns.histplot(data[col],
                                  color='green', stat="density")
                sns.kdeplot(data[col], ax=ax, color='red',
                            shade=True)
                st.pyplot(plt)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def null_dist(self, data, explode):
        """this function showing null distribution columns"""
        try:
            data_null_count = data.isna().sum()
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.title("Based On Whole Data")
            plt.pie(x=[data.shape[0], data_null_count.sum()],
                    labels=["Data", "Null"],
                    autopct="%.2f%%")
            plt.subplot(2, 2, 2)
            sns.barplot(y=[data.shape[0], data_null_count.sum()],
                        x=["Data", "Null"])
            plt.subplot(2, 2, 3)
            plt.title("Based On Specific Features")
            plt.pie(x=data_null_count[data_null_count != 0],
                    labels=data_null_count[
                        data_null_count != 0].index,
                    autopct="%.2f%%",
                    explode=[explode] * len(
                        data_null_count[data_null_count != 0]))
            plt.subplot(2, 2, 4)
            sns.barplot(y=data_null_count[data_null_count != 0],
                        x=data_null_count[
                        data_null_count != 0].index)
            st.pyplot(plt)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def show_corr(self, data):
        """this function showing correlation of data"""
        try:
            data_copy = data.select_dtypes(exclude=['object'])
            mask = np.triu(np.ones_like(data_copy.corr()))
            plt.figure(figsize=(15, 10))
            sns.heatmap(data_copy.corr(), mask=mask,
                        cmap="hot", annot=True)
            st.pyplot(plt)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None

    def feature_relation(self, data, feature1,
                         feature2, based_feature=None):
        """this function showing relation between 2 feature,
        can be based on third feature"""
        try:
            if based_feature:
                plt.figure(figsize=(15, 10))
                sns.scatterplot(x=data[feature1],
                                y=data[feature2],
                                hue=data[based_feature])
                st.pyplot(plt)
            else:
                plt.figure(figsize=(15, 10))
                sns.scatterplot(x=data[feature1],
                                y=data[feature2])
                st.pyplot(plt)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None
