

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
from scripts.utils import *
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Setting up all directory paths
root_folder = "/opt/airflow"
old_data_directory = os.path.join(root_folder, "data/raw/")
new_data_directory = os.path.join(root_folder, "data/new/")

# Database configurations
db_path = os.path.join(root_folder, "database/")
db_file_name = "feature_store_v01.db"
drfit_db_name = "drift_db_name.db"
date_columns = [
    'registration_init_time', 
    'transaction_date_min', 
    'transaction_date_max', 
    'membership_expire_date_max', 
    'last_login'
]

# MLflow configurations
mlflow_serving_uri = "http://mlflow_serve:5000/invocations"
run_on = "new"  # Specify whether to run on new or old data
append = False
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'

# Default arguments for the DAG
default_args = {
    'owner': 'CharlieThomas',              # The owner of the DAG
    'depends_on_past': False,              # Indicates whether the task should depend on previous runs
    'start_date': days_ago(2),             # Set the start date to 2 days ago
    'provide_context': True                 # Allow tasks to access contextual information
}

# Create a new DAG for the inference pipeline
dag = DAG(
    'Inference',                           # Name of the DAG
    default_args=default_args,             # Default arguments specified above
    schedule_interval='0 0 * * 2',        # Schedule the DAG to run every Tuesday at midnight
    max_active_runs=1,                     # Allow only one active run of this DAG at a time
    tags=['inference_pipeline']             # Tags for organizing DAGs in the Airflow UI
)

# Task to reset process flags in the database
op_reset_processes_flags = PythonOperator(
    task_id='reset_processes_flag',
    python_callable=get_flush_db_process_flags,
    op_kwargs={'db_path': db_path, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Reset process flags in the drift database to ensure a fresh start for the inference pipeline.
          This task is executed at the beginning of each DAG run."""
)

# Task to create the necessary databases
op_create_db = PythonOperator(
    task_id='create_check_db', 
    python_callable=build_dbs,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
    dag=dag,
    doc="""Create the necessary databases for storing features and drift metrics.
          This task ensures the databases are in place before data processing starts."""
)

# Task to load data from the specified source
op_load_data = PythonOperator(
    task_id='load_data', 
    python_callable=load_data_from_source,
    op_kwargs={
        'db_path': db_path,
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name,
        'old_data_directory': old_data_directory,
        'new_data_directory': new_data_directory,
        'run_on': run_on,
        'start_date': start_date,
        'end_date': end_date,
        'append': append
    },
    dag=dag,
    doc="""Load data from the specified source directories into the database.
          This task is responsible for populating the database with the most recent data."""
)

# Task to process transaction data
op_process_transactions = PythonOperator(
    task_id='process_transactions',
    python_callable=get_membership_data_transform,
    op_kwargs={
        'db_path': db_path, 
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name
    },
    dag=dag,
    doc="""Process transaction data to transform it into a suitable format for analysis.
          This task applies necessary transformations and prepares the data for further processing."""
)

# Task to process membership data
op_process_members = PythonOperator(
    task_id='process_members', 
    python_callable=get_transaction_data_transform,
    op_kwargs={
        'db_path': db_path, 
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name
    },
    dag=dag,
    doc="""Process membership data to prepare it for modeling.
          This task handles the transformation and validation of membership-related features."""
)

# Task to process user logs
op_process_userlogs = PythonOperator(
    task_id='process_userlogs',
    python_callable=get_user_data_transform,
    op_kwargs={
        'db_path': db_path, 
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name
    },
    dag=dag,
    doc="""Process user log data to derive useful insights for model predictions.
          This task focuses on cleaning and structuring user log information."""
)

# Task to merge processed data
op_merge = PythonOperator(
    task_id='merge_data',
    python_callable=get_final_data_merge,
    op_kwargs={
        'db_path': db_path, 
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name
    },
    dag=dag,
    doc="""Merge all processed data into a final dataset for modeling.
          This task consolidates the outputs of various processing tasks into a single dataset."""
)

# Task to prepare data for modeling
op_process_data = PythonOperator(
    task_id='data_preparation', 
    python_callable=get_data_prepared_for_modeling,
    op_kwargs={
        'db_path': db_path,
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name,
        'date_columns': date_columns,
        'date_transformation': date_transformation
    },
    dag=dag,
    doc="""Prepare the merged dataset for modeling by performing necessary transformations.
          This task includes handling date columns and applying any required preprocessing steps."""
)

# Task to predict using MLflow server
op_predict_data = PythonOperator(
    task_id='Prediction', 
    python_callable=get_predict_mlflow_server,
    op_kwargs={
        'db_path': db_path, 
        'db_file_name': db_file_name,
        'drift_db_name': drfit_db_name
    },
    dag=dag,
    doc="""Use the MLflow server to make predictions based on the prepared data.
          This task sends the prepared dataset to the model and retrieves the predictions."""
)

# Task to send email notification upon completion
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email = EmailOperator(
    task_id='send_email', 
    to='charliethomasctg@gmail.com', 
    subject='Inference Pipeline Execution Finished', 
    html_content=f"The execution of Inference pipeline is finished @ {timestamp}. Check Airflow Logs for more details or check SQLite Backend for Predictions.", 
    dag=dag,
    doc="""Send an email notification upon the successful completion of the inference pipeline.
          This task informs stakeholders of the execution results and directs them to logs or the SQLite backend."""
)

# Set the task sequence using the >> operator for better readability
op_reset_processes_flags >> op_create_db >> op_load_data
op_load_data >> [op_process_members, op_process_userlogs, op_process_transactions]
op_process_members >> op_merge
op_process_userlogs >> op_merge
op_process_transactions >> op_merge
op_merge >> op_process_data >> op_predict_data >> send_email
