import streamlit as st
import pandas as pd
from dataanalysis import EDA
from dataprepartion import DataPrepare
from models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss, \
    mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Problem Solver",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

eda = EDA()
data_prep = DataPrepare()
select_model = Model()


# read data csv file
def read_data(file):
    """this function read csv file"""
    data = pd.read_csv(file)
    return data


def main():
    # remove hamburger and footer
    st.markdown("""
    <style>
    .css-1rs6os.edgvbvh3
    {
      visibility:hidden;
    }
    .css-10pw50.egzxvld1
    {
      visibility:hidden;
    }
    <style/>
    """, unsafe_allow_html=True)
    if "train_data" not in st.session_state:
        st.session_state.train_data = None
        st.session_state.test_data = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "select_output" not in st.session_state:
        st.session_state.select_output = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
        st.session_state.X_val = None
        st.session_state.y_train = None
        st.session_state.y_val = None
    st.title("ML Problem solver")
    with st.form("Upload Train Data"):
        st.header("Upload Data")
        ud_col_1, ud_col_2 = st.columns(2)
        train_file = ud_col_1.file_uploader("Upload train data file:", type="csv")
        test_file = ud_col_2.file_uploader("Upload test data file:", type="csv")
        proc = st.form_submit_button("process")
    ud_col_1, ud_col_2, ud_col_3, ud_col_4 = st.columns(4)
    if train_file:
        show_data = ud_col_1.checkbox("Show Data")
        show_shape = ud_col_2.checkbox("Show Shape")
        show_cols = ud_col_3.checkbox("Show Columns`s Name")
        show_dis = ud_col_4.checkbox("Show Description")
        if proc:
            with st.spinner():
                st.session_state.train_data = read_data(train_file)
                if test_file:
                    st.session_state.test_data = read_data(test_file)
        if show_data:
            st.subheader("Data")
            st.dataframe(st.session_state.train_data)
            if test_file:
                st.dataframe(st.session_state.test_data)
        if show_shape:
            st.subheader("Shape")
            st.write(st.session_state.train_data.shape)
            if test_file:
                st.write(st.session_state.test_data.shape)
        if show_cols:
            st.subheader("Columns`s Name")
            st.write(st.session_state.train_data.columns)
            if test_file:
                st.write(st.session_state.test_data.columns)
        if show_dis:
            st.subheader("Description")
            st.write(st.session_state.train_data.describe())
            if test_file:
                st.write(st.session_state.test_data.describe())

        st.header("Data Analysis")
        da_col_1, da_col_2, da_col_3, da_col_4 = st.columns(4)
        show_data_dist = da_col_1.checkbox("Show Data Distribution")
        show_null_dist = da_col_2.checkbox("Show Null Distribution")
        show_correlation = da_col_3.checkbox("Show Correlation")
        show_relation = da_col_4.checkbox("Show Relation of Feature")
        if show_data_dist:
            st.subheader("Data Distribution")
            colm = st.selectbox("Choose Column:",
                                st.session_state.train_data.columns)
            w_figure = st.slider("Choose Figure Width:",
                                 min_value=1, max_value=50,
                                 value=15)
            h_figure = st.slider("Choose Figure Height:",
                                 min_value=1, max_value=50,
                                 value=10)
            action = st.button("Show Figure")
            if action:
                eda.data_dist(st.session_state.train_data, colm,
                              fig_size=(w_figure, h_figure))
                if test_file:
                    eda.data_dist(st.session_state.test_data,
                                  st.session_state.test_data.columns,
                                  fig_size=(w_figure, h_figure))
        if show_null_dist:
            st.subheader("Null Distribution")
            exp = st.slider("Determine Explode:",
                            min_value=0.0, max_value=1.0,
                            step=0.01)
            eda.null_dist(st.session_state.train_data, exp)
            if test_file:
                eda.null_dist(st.session_state.test_data, exp)
        if show_correlation:
            st.subheader("Features Correlation")
            eda.show_corr(st.session_state.train_data)
            if test_file:
                eda.show_corr(st.session_state.test_data)
        if show_relation:
            st.subheader("Feature Relation")
            feature_1 = st.selectbox("Select The First Feature:",
                                     st.session_state.train_data.select_dtypes(exclude=['object']).columns)
            feature_2 = st.selectbox("Select The Second Feature:",
                                     st.session_state.train_data.select_dtypes(exclude=['object']).columns)
            target_based = st.selectbox("Based Feature:",
                                        st.session_state.train_data.select_dtypes(exclude=['object']).columns)
            based_action = st.checkbox("Press Here If you want Based Feature")
            action = st.button("Show Relation", )
            if action:
                test_data = st.session_state.test_data
                if based_action:
                    eda.feature_relation(st.session_state.train_data,
                                         feature_1,
                                         feature_2, target_based)
                    if test_file:
                        if test_data[feature_1] and test_data[feature_2] and test_data[target_based]:
                            eda.feature_relation(st.session_state.test_data,
                                                 feature_1,
                                                 feature_2, target_based)
                else:
                    eda.feature_relation(st.session_state.train_data,
                                         feature_1,
                                         feature_2)
                    if test_file:
                        if test_data[feature_1] and test_data[feature_2]:
                            eda.feature_relation(st.session_state.test_data,
                                                 feature_1,
                                                 feature_2)

        st.header("Data Preparation")
        if st.session_state.train_data.isna().sum().sum():
            clean_nulls = st.checkbox("Clean nulls")
            if clean_nulls:
                st.subheader("Clean Train Data")

                object_data, numeric_data = st.session_state.train_data.select_dtypes(
                    exclude=[int, float]
                ) \
                    , st.session_state.train_data.select_dtypes(
                    exclude=object
                )
                object_cols = object_data.isna().sum()[object_data.isna().sum() != 0].index
                numeric_cols = numeric_data.isna().sum()[numeric_data.isna().sum() != 0].index
                if st.session_state.train_data.isna().sum().sum():
                    mean_cols = st.multiselect("Choose columns that you want to fill with mean:",
                                               [col for col in numeric_cols])
                    mode_cols = st.multiselect("Choose columns that you want to fill with mode:",
                                               [col for col in numeric_cols if
                                                col not in mean_cols])
                    combined_types = object_cols.tolist() + numeric_cols.tolist()
                    ffill_cols = st.multiselect("Choose columns that you want to fill forward:",
                                                [col for col in combined_types if
                                                 col not in mean_cols and col not in mode_cols])
                    bfill_cols = st.multiselect("Choose columns that you want to fill backward:",
                                                [col for col in combined_types if
                                                 col not in mean_cols and col not in mode_cols
                                                 and col not in ffill_cols])
                    specific_value = st.selectbox("Choose columns that you want to fill specific value:",
                                                  ["Choose an option"] + [col for col in combined_types if col not in
                                                                          mean_cols and col not in mode_cols and
                                                                          col not in ffill_cols and
                                                                          col not in bfill_cols])

                    action = st.button("Perform")
                    if action:
                        if mean_cols:
                            data_prep.fill_mean(st.session_state.train_data,
                                                mean_cols)
                        if mode_cols:
                            data_prep.fill_mode(st.session_state.train_data,
                                                mode_cols)
                        if ffill_cols:
                            data_prep.fill_forward(st.session_state.train_data,
                                                   ffill_cols)
                        if bfill_cols:
                            data_prep.fill_backward(st.session_state.train_data,
                                                    bfill_cols)
                        if specific_value:
                            if specific_value in numeric_cols:
                                val = int(st.text_input("Enter the value", max_chars=32))
                                data_prep.fill_value(st.session_state.train_data,
                                                     mode_cols,
                                                     val)
                            elif specific_value in object_cols:
                                val = st.text_input("Enter the value", max_chars=32)
                                data_prep.fill_value(st.session_state.train_data,
                                                     mode_cols,
                                                     val)

        remove_cols = st.multiselect("Choose columns that you want to remove:",
                                     st.session_state.train_data.columns)
        delete = st.button("Delete")
        if remove_cols and delete:
            data_prep.remove_col(st.session_state.train_data,
                                 remove_cols)
            if test_file:
                data_prep.remove_col(st.session_state.test_data,
                                     remove_cols)
        dp_col_1, dp_col_2, dp_col_3 = st.columns(3)
        scale = dp_col_1.checkbox("Scale Data")
        if scale:
            st.subheader("Scale Data")
            scaling_cols = st.multiselect("Choose Columns that you want to scale",
                                          st.session_state.train_data.select_dtypes(exclude=[object]).columns)
            scaling = st.button("Scaling")
            if scaling_cols and scaling:
                st.session_state.train_data[scaling_cols] = data_prep.scaler(st.session_state.train_data,
                                                                             scaling_cols)
                if test_file:
                    st.session_state.test_data[scaling_cols] = data_prep.scaler(st.session_state.test_data,
                                                                                scaling_cols)

        convert_cat_num = dp_col_2.button("Convert Categorical to numerical")
        if convert_cat_num:
            st.session_state.train_data = data_prep.get_dummy(
                st.session_state.train_data,
                st.session_state.train_data.select_dtypes(exclude=[float, int]).columns)
            if test_file:
                st.session_state.test_data = data_prep.get_dummy(
                    st.session_state.test_data,
                    st.session_state.test_data.select_dtypes(exclude=[float, int]).columns)
        split = dp_col_3.checkbox("Split Data")
        if split:
            st.subheader("Split Data")
            st.session_state.select_output = st.selectbox("Choose which column is output:",
                                                          st.session_state.train_data.columns)
            train_sz = st.slider("Choose train size:",
                                 min_value=0.1,
                                 max_value=1.0,
                                 step=0.1,
                                 value=0.7)
            split = st.button("Split")
            if split and st.session_state.select_output:
                st.subheader("Split Data")
                st.session_state.X_train, st.session_state.X_val, \
                    st.session_state.y_train, \
                    st.session_state.y_val = data_prep.select_data(
                    st.session_state.train_data,
                    st.session_state.select_output,
                    train_sz)
                st.session_state.y_val = st.session_state.y_val.to_frame()
                st.session_state.y_val.reset_index(inplace=True)
                st.session_state.y_val = st.session_state.y_val.iloc[:, -1]
                st.write(f"X_train shape = {st.session_state.X_train.shape}")
                st.write(f"X_val shape = {st.session_state.X_val.shape}")
                st.write(f"y_train shape = {st.session_state.y_train.shape}")
                st.write(f"y_val shape = {st.session_state.y_val.shape}")

        st.header("Model")
        type_problem = st.selectbox("Choose Problem:",
                                    ["Classification",
                                     "Regression"])
        if type_problem == "Classification":
            model_text = st.selectbox("Choose Model:",
                                      ["LGBMClassifier",
                                       "XGBClassifier",
                                       "RandomForestClassifier"])
            n_est = st.slider("Choose number of estimating:",
                              min_value=10,
                              value=100,
                              max_value=2000)
            train = st.button("Train")
            if train:
                if model_text == "LGBMClassifier":

                    st.session_state.model = select_model.lgbmc(st.session_state.X_train,
                                                                st.session_state.y_train,
                                                                n_est=n_est)
                elif model_text == "XGBClassifier":

                    st.session_state.model = select_model.xgbc(st.session_state.X_train,
                                                               st.session_state.y_train,
                                                               n_est=n_est)
                else:

                    st.session_state.model = select_model.rfc(st.session_state.X_train,
                                                              st.session_state.y_train,
                                                              n_est=n_est)
        else:
            model_text = st.selectbox("Choose Model:",
                                      ["LGBMRegressor",
                                       "XGBRegressor",
                                       "RandomForestRegressor"])
            n_est = st.slider("Choose number of estimating:",
                              min_value=10,
                              value=100,
                              max_value=2000)
            train = st.button("Train")
            if train:
                if model_text == "LGBMRegressor":

                    st.session_state.model = select_model.lgbmr(st.session_state.X_train,
                                                                n_est=n_est)
                elif model_text == "XGBRegressor":

                    st.session_state.model = select_model.xgbr(st.session_state.X_train,
                                                               st.session_state.y_train,
                                                               n_est=n_est)
                else:

                    st.session_state.model = select_model.rfr(st.session_state.X_train,
                                                              st.session_state.y_train,
                                                              n_est=n_est)

        if st.session_state.model:
            show_pred_data = st.checkbox("Show Predict Data")
            y_pred = st.session_state.model.predict(
                st.session_state.X_val
            )
            st.session_state.y_pred = pd.DataFrame(y_pred, columns=["Predict Values"])

            if show_pred_data:
                combined_df = pd.concat([st.session_state.y_val,
                                         st.session_state.y_pred], axis=1)
                st.write(combined_df)

            st.header("Evaluation")
            if type_problem == "Classification":
                st.write("Based on this counts Enter labels")
                st.write(st.session_state.train_data[st.session_state.select_output].value_counts().keys())
                labels = []
                len_labels = len(
                    st.session_state.y_val.value_counts())
                i = 1
                while len_labels:
                    label = st.text_input(f"Enter label_{i}:",
                                          max_chars=20)
                    labels.append(label)
                    len_labels -= 1
                    i += 1
                show_class_report = st.checkbox("Show Classification Report")
                if show_class_report:
                    class_report = classification_report(
                        y_true=st.session_state.y_val,
                        y_pred=st.session_state.y_pred)
                    st.text(class_report)
                    if i - 1 == 2:
                        loss = log_loss(y_true=st.session_state.y_val,
                                        y_pred=st.session_state.y_pred)
                        st.write(f"Binary Classification Loss = {round(loss, 2)}")
                show_cm = st.checkbox("Show Confusion Matrix")
                cm = confusion_matrix(
                    y_true=st.session_state.y_val,
                    y_pred=st.session_state.y_pred)
                if show_cm:
                    plt.figure(figsize=(15, 10))
                    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
                    st.pyplot(plt)
            else:
                show_pred_diff = st.checkbox("Show Difference between predicted and actual value")
                if show_pred_diff:
                    plt.figure(figsize=(15, 10))
                    plt.plot(np.array(st.session_state.y_val).flatten(), color="green")
                    plt.plot(np.array(st.session_state.y_pred).flatten(), color="red")
                    st.pyplot(plt)
                loss_menu = st.selectbox("Choose loss metric",
                                         ["MAE", "MSE"])
                if loss_menu == "MAE":
                    loss = mean_absolute_error(y_true=st.session_state.y_val,
                                               y_pred=st.session_state.y_pred)
                    st.text(f"MAE loss = {loss}")
                elif loss_menu == "MES":
                    loss = mean_squared_error(y_true=st.session_state.y_val,
                                              y_pred=st.session_state.y_pred)
                    st.text(f"MSE loss = {loss}")

            if test_file:
                pred_test = st.checkbox("Predict Test Data")
                if pred_test:
                    test_pred = pd.DataFrame(st.session_state.model.predict(
                        st.session_state.test_data
                    )).rename(columns={0: "Predicted Test Data"})

                    st.write(test_pred)


if __name__ == "__main__":
    main()
