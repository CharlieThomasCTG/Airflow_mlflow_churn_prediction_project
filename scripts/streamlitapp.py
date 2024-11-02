import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import os
import sqlite3
from sklearn.preprocessing import StandardScaler
from utils import *

# Database configuration
root_folder = "/opt/airflow"
db_file_name = "db_final_name.db"
db_path = os.path.join(root_folder, "database")


# Airflow API configuration
AIRFLOW_URL = "http://churn-airflow:8085/api/v1/dags/{dag_id}/dagRuns"  # Update with your Airflow setup

# Function to trigger a DAG
def trigger_dag(dag_id):
    url = AIRFLOW_URL.format(dag_id=dag_id)
    response = requests.post(url, auth=('Admin', 'BavCzWn8dGWyyFRg'))  # Replace with your Airflow username and password
    
    if response.status_code == 200:
        st.success(f"DAG '{dag_id}' triggered successfully!")
    else:
        st.error(f"Failed to trigger DAG '{dag_id}': {response.text}")

# Streamlit UI
st.sidebar.header("Airflow Management")
dag_id = st.sidebar.selectbox("Select DAG to Trigger", ["Data_End2End_Processing","Model_Building_Pipeline", "Drift_Pipeline", "Inference"])  # Replace with your DAG IDs

if st.sidebar.button("Trigger DAG"):
    trigger_dag(dag_id)

# Create SQLite connection
def create_sqlit_connection(db_path, db_file):
    full_db_path = os.path.join(db_path, db_file)
    if not os.path.exists(full_db_path):
        st.error("Database file does not exist.")
        return None
    return sqlite3.connect(full_db_path)

# Streamlit App Configuration
st.sidebar.header("Machine Learning Management App")
option = st.sidebar.selectbox("Choose an option", ["Make Prediction", "View Predictions"])

# Configuration Inputs
MLFLOW_URL = "http://mlflow_serve:5000/invocations"  # Update this to the serving endpoint in Docker

date_columns = ['registration_init_time', 'transaction_date_min', 'transaction_date_max', 'membership_expire_date_max', 'last_login']

# Function to make predictions using the MLflow served model
def make_prediction():
    st.header("Make Predictions")
    st.write("Upload a new dataset for predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        newdata = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.dataframe(newdata.head())
        
        new_data = unseen_data_preparation(newdata, scale_method='standard', date_columns=None, corr_threshold=0.90, drop_corr=False,
                                            date_transformation=True)
        st.success("Transformed Data:")
        st.dataframe(new_data)

        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    # Convert the DataFrame to JSON format for MLflow request
                    data_json = new_data.to_json(orient="split")

                    # Send request to MLflow model endpoint
                    response = requests.post(
                        MLFLOW_URL,
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"dataframe_split": json.loads(data_json)})
                    )

                    # Check if the request was successful
                    if response.status_code == 200:
                        predictions = response.json()
                        raw_predictions = predictions.get("predictions", [])
                        
                        # Clean and structure the data
                        clean_predictions = [int(item.split(" - ")[0]) if isinstance(item, str) else item for item in raw_predictions]
                        predictions_df = pd.DataFrame(clean_predictions, columns=['Predictions'])
                        st.success("Prediction completed successfully!")
                        st.write("Prediction Results:")
                       
                        if len(predictions_df) == len(newdata):
                            newdata['Predictions'] = predictions_df['Predictions']
                            

                            # Write predictions to SQLite database
                            try:
                                conn = create_sqlit_connection(db_path, db_file_name)
                                if conn is not None:
                                    newdata.to_sql('prediction_table', conn, if_exists='append', index=False)
                                    st.success("Predictions written to the database successfully!")
                                    conn.close()
                            except Exception as e:
                                st.error(f"Error writing predictions to database: {e}")
                        else:
                            st.error("The number of rows in predictions_df and newdata do not match.")
                    else:
                        st.error(f"Error during prediction: {response.text}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Function to view predictions from the database
def view_final_predictions(db_path, db_file_name):
    st.header("View Final Predictions")
    
    try:
        conn = create_sqlit_connection(db_path, db_file_name)
        if conn is not None:
            query = 'SELECT msno as membership_number, predictions as is_churn FROM prediction_table'
            final_predictions_df = pd.read_sql(query, conn)

            if final_predictions_df.empty:
                st.warning("No predictions found in the database.")
            else:
                st.write("Final Predictions:")
                st.dataframe(final_predictions_df)

            conn.close()

    except Exception as e:
        st.error(f"Error retrieving final predictions: {e}")

# Main Section
if option == "Make Prediction":
    make_prediction()
elif option == "View Predictions":
    view_final_predictions(db_path, db_file_name)
