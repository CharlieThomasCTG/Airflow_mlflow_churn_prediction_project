import warnings
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from airflow.operators.email_operator import EmailOperator
import mlflow
from scripts.utils import *

warnings.filterwarnings('ignore')

# Path configurations
root_folder = "/opt/airflow"
old_data_directory = root_folder + "/data/raw/"
new_data_directory = root_folder + "/data/new/"

# Database configuration
db_path = root_folder + "/database/"
db_file_name = "feature_store_v01.db"
drfit_db_name = "drift_db_name.db"
date_columns = ['registration_init_time', 'transaction_date_min', 'transaction_date_max',
                'membership_expire_date_max', 'last_login']

# MLflow settings
run_on = "old"  # Specify whether to run on old or new data
append = False
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'

# Set the tracking URI to the MLflow service
mlflow.set_tracking_uri("http://mlflow:6006")

# Declare default arguments for the DAG
default_args = {
    'owner': 'CharlieThomas',              # Owner of the DAG
    'depends_on_past': False,              # Whether this task should depend on the previous task's state
    'start_date': days_ago(2),             # Start the DAG from 2 days ago
    'provide_context': True                 # Provide context to tasks for additional information
}

# Creating a new DAG for the model building pipeline
dag = DAG(
    'Model_Building_Pipeline',              # Name of the DAG
    default_args=default_args,               # Use the previously defined default arguments
    schedule_interval='0 0 * * 2',          # Schedule to run every Tuesday at midnight
    max_active_runs=1,                       # Limit to one active run at a time
    tags=['ml_pipeline']                     # Tag for organizing this DAG in the Airflow UI
)

# Task to reset process flags in the database
op_reset_processes_flags = PythonOperator(
    task_id='reset_processes_flag',
    python_callable=get_flush_db_process_flags,
    op_kwargs={'db_path': db_path, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Reset process flags in the drift database to ensure a fresh start for the model building pipeline.
          This task is executed at the beginning of each DAG run."""
)

# Task to create the database for drift checks
op_create_db = PythonOperator(
    task_id='create_check_db',
    python_callable=build_dbs,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
    dag=dag,
    doc="""Create the necessary database for performing drift checks.
          This task ensures that the drift database is set up before any data processing occurs."""
)

# Task to load data from the source
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
        'end_date': end_date
    },
    dag=dag,
    doc="""Load data from specified source directories into the database.
          This task is crucial for ensuring that the most recent data is available for processing."""
)

# Task to process membership data
op_process_members = PythonOperator(
    task_id='process_members',
    python_callable=get_membership_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Transform membership data to prepare it for modeling.
          This task handles data cleaning and structuring for further analysis."""
)

# Task to process transaction data
op_process_transactions = PythonOperator(
    task_id='process_transactions',
    python_callable=get_transaction_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Transform transaction data to make it suitable for analysis.
          This task applies necessary transformations and prepares the transaction data for integration."""
)

# Task to process user logs
op_process_userlogs = PythonOperator(
    task_id='process_userlogs',
    python_callable=get_user_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Transform user log data to extract meaningful features.
          This task focuses on cleaning and preparing user logs for downstream processing."""
)

# Task to merge processed data
op_merge = PythonOperator(
    task_id='merge_data',
    python_callable=get_final_data_merge,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Merge all processed datasets into a single final dataset for modeling.
          This task consolidates the outputs from various processing tasks."""
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
    doc="""Prepare the merged dataset for model training by performing necessary transformations.
          This task includes handling date columns and other preprocessing steps."""
)

# Task for model training with hyperparameter tuning
op_model_training_with_tuning = PythonOperator(
    task_id='Model_Training_hpTunning',
    python_callable=get_train_model_hptune,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Train the model using the prepared dataset, including hyperparameter tuning.
          This task is crucial for optimizing model performance."""
)

# Email notification task upon completion
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email = EmailOperator(
    task_id='send_email',
    to='charliethomasctg@gmail.com',
    subject='Model Building Pipeline Execution Finished',
    html_content=f"The execution of Model Building pipeline is finished @ {timestamp}. Check Airflow Logs for more details or check MLFLOW for Final Model",
    dag=dag,
    doc="""Send an email notification upon the successful completion of the model building pipeline.
          This task informs stakeholders of the execution results and directs them to relevant logs or model details."""
)

# Set the task sequence using the >> operator
op_reset_processes_flags >> op_create_db >> op_load_data
op_load_data >> [op_process_members, op_process_userlogs, op_process_transactions]
op_process_members >> op_merge
op_process_userlogs >> op_merge
op_process_transactions >> op_merge
op_merge >> op_process_data >> op_model_training_with_tuning >> send_email

