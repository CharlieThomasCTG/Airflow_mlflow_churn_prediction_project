from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from scripts.utils import *

# Setting up all directory paths and configuration variables
root_folder = "/opt/airflow"
old_data_directory = root_folder+"/data/raw/"
new_data_directory = root_folder+"/data/new/"
db_file_name = "feature_store_v01.db"
drfit_db_name = "drift_db_name.db"
db_path = root_folder+"/database/"
date_columns = ['registration_init_time', 'transaction_date_min', 'transaction_date_max', 'membership_expire_date_max', 'last_login']
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'
run_on = "old"

# Default arguments for the DAG
default_args = {
    'owner': 'CharlieThomas',          # Owner of the DAG, used for tracking
    'depends_on_past': False,          # If set to True, task instances will not run unless the previous task instance succeeded
    'start_date': days_ago(2),         # The start date for the DAG, using Airflow's utility to specify a date relative to today
    'provide_context': True,            # If set to True, task instances can access the context dictionary for additional information
}


# Creating a new DAG for end-to-end data processing
dag = DAG(
    dag_id='Data_End2End_Processing',  # Unique identifier for the DAG
    default_args=default_args,          # Default arguments for tasks within the DAG
    schedule_interval='0 0 * * 2',      # Schedule the DAG to run at midnight every Tuesday
    max_active_runs=1,                  # Limit to one active run at a time
    tags=['data_pipeline']               # Tag for categorizing the DAG
)


# Integrating different operator tasks in the Airflow DAG

op_reset_processes_flags = PythonOperator(
    task_id='reset_processes_flag',
    python_callable=get_flush_db_process_flags,
    op_kwargs={'db_path': db_path, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Reset processes flags in the drift database to ensure a fresh start for the data pipeline.
        This task is executed at the beginning of each DAG run."""
)

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
    },
    dag=dag,
    doc="""Load data from the source directories based on specified start and end dates. 
        This task loads raw data for further processing and transformation."""
)

op_process_members = PythonOperator(
    task_id='process_members', 
    python_callable=get_membership_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Process membership data by applying necessary transformations and storing the results.
        This includes cleaning and preparing data for member information."""
)

op_process_transactions = PythonOperator(
    task_id='process_transactions',
    python_callable=get_transaction_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Process transaction data by transforming and preparing the dataset for integration.
        This task handles data such as transaction amounts, dates, and other related information."""
)

op_process_userlogs = PythonOperator(
    task_id='process_userlogs',
    python_callable=get_user_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Process user logs to capture user activity data, transforming them for further analysis.
        This includes preparing log data to support user behavior insights."""
)

op_merge = PythonOperator(
    task_id='merge_data',
    python_callable=get_final_data_merge,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Merge processed data from membership, transaction, user logs and churn logs into a final dataset.
        This unified dataset will be used in downstream data processing tasks."""
)

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
    doc="""Prepare data for modeling by applying transformations such as date handling and feature engineering.
        This includes processing date columns and other pre-modeling data preparations."""
)

# Timestamp for email notification
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email = EmailOperator(
    task_id='send_email', 
    to='charliethomasctg@gmail.com', 
    subject='Data Pipeline Execution Finished', 
    html_content=f"The execution of the data pipeline is finished @ {timestamp}. Check Airflow logs for more details or check SQLite backend for transformed data.", 
    dag=dag,
    doc="""Send an email notification upon completion of the data pipeline run."""
)

# Set the task sequence using the bitwise right shift operator
op_reset_processes_flags >> op_load_data
op_load_data >> [op_process_members, op_process_userlogs, op_process_transactions]
op_process_members >> op_merge
op_process_userlogs >> op_merge
op_process_transactions >> op_merge
op_merge >> op_process_data >> send_email

