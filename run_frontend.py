"""
Runs the streamlit front end for the mlexec demo.
"""
import pandas as pd
import streamlit as st
from mlexec import MLExecutor

MODEL_OPTIONS = {"lgb","svm","nn","lr","rf","xgb","knn"}
TASK_OPTIONS = {"classification","regression"}
METRIC_OPTIONS = {"classification": ["matthews_corrcoef","misclassification_cost",
                    "accuracy", "recall", "precision","f1_score","auc_roc","auc_pr"],
                "regression": ["rmse","r2","mae","mape"]}
CV_OPTIONS = ["basic","nested",""]

def run_mlexec(df_: pd.DataFrame,
                target_col:str,
                task:str="classification",
                model_list:list=["lgb","rf","xgb"],
                metric:str="",
                excluded_cols:list=[],
                n_fold:int=4,
                cv_type:str="basic",
                num_config:int=10):
    """Running MLexecutor using the input paramters provided by the user.

    Args:
        df_ (pd.DataFrame): The dataframe containing the dependent and independent variables. \n
        target_col (str): The column which contains the value for the dependent variable. \n
        task (str, optional): The task: `"regression"` or `"classification"`.\n
        model_list (list, optional): List of models to tune. Defaults to ["lgb","rf","xgb"].\n
        metric (str, optional): The metric with which the model should \
        be compared.\n
        excluded_cols (list, optional): Columns which should not be used \
        in the modelling process.\n
        n_fold (int, optional): Number of folds to be used in k-fold cross \
        validation.\n
        cv_type (str, optional): Type of cross validation to perform.\n
        num_config (int, optional): Number of model configurations to run during tuning.
    """
    mle = MLExecutor(
        df_,
        target_col=target_col,
        task=task,
        model_list=model_list,
        metric=metric,
        exclude_cols=excluded_cols,
        cv=cv_type,
        n_fold=n_fold,
        max_evals=num_config,
    )
    st.write("Models executed successfully!")
    st.balloons()
    st.dataframe(mle.val_results)
    st.dataframe(mle.test_results)


def take_user_inputs():
    """
    Obtain user inputs to be passed to the MLExec object
    """
    uploaded_file = st.file_uploader("Choose a tabular file to begin")
    if uploaded_file:
        ## Reading as a pandas dataframe based on format
        suffix = uploaded_file.name.split(".")[-1]
        if suffix in ["xlsx","xls"]:
            df_ = pd.read_excel(uploaded_file)
        elif suffix in ["csv","txt","data"]:
            df_ = pd.read_csv(uploaded_file)
        elif suffix in ["pkl"]:
            df_ = pd.read_pickle(uploaded_file)
        elif suffix in ["json"]:
            df_ = pd.read_json(uploaded_file)

        col_list = list(df_.columns)
        target_col = st.selectbox("Please select the target column",
                                col_list)
        excluded_cols = st.multiselect("please select columns to exclude (if any)",
                                col_list,
                                default=[])
    task = st.selectbox("Please select ML task",
                TASK_OPTIONS)
    model_list = st.multiselect("Please select models",
                MODEL_OPTIONS,
                default=["lgb","rf","xgb"])
    metric = st.selectbox("Please select a metric for model comparison",
                METRIC_OPTIONS[task])
    cv_type = st.selectbox("Please cross validation method",
                CV_OPTIONS)
    if cv_type:
        n_fold = st.slider('Number of k-fold in CV', 1, 10, 2)

    num_config = st.slider('Number of model configurations to tune',
                           5, 50, 5)
    response = st.button("Click to begin tuning")

    if response:
        return {"df_": df_,
                "target_col": target_col,
                "excluded_cols": excluded_cols,
                "task": task,
                "model_list": model_list,
                "metric": metric,
                "cv_type": cv_type,
                "n_fold": n_fold,
                "num_config": num_config}

def main():
    """
    Main function to handle user inputs, ML execution and displaying output
    """
    user_inputs = take_user_inputs()

    if user_inputs:
        run_mlexec(**user_inputs)

main()
