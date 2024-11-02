# Building the DAG using the functions from data_process and model module

from scripts.utils import *
import warnings
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
import mlflow

# Suppress warnings
warnings.filterwarnings('ignore')

# Setting up all directory paths
root_folder = "/opt/airflow"
old_data_directory = os.path.join(root_folder, "data/raw/")
new_data_directory = os.path.join(root_folder, "data/new/")

# Database configuration
db_path = os.path.join(root_folder, "database/")
db_file_name = "feature_store_v01.db"
drfit_db_name = "drift_db_name.db"
date_columns = ['registration_init_time', 'transaction_date_min', 'transaction_date_max',
                'membership_expire_date_max', 'last_login']
drift_db_name = "drift_db_name.db"

# Processing parameters
run_on = "old"  # Indicates which dataset to use
start_date = '2017-03-01'
end_date = '2017-03-31'
metric = 'std'  # Metric for drift detection

# Set the tracking URI for MLflow
mlflow.set_tracking_uri("http://mlflow:6006")

# Default arguments for the DAG
default_args = {
    'owner': 'CharlieThomas',              # The owner of the DAG
    'depends_on_past': False,              # Do not depend on past DAG runs
    'start_date': days_ago(2),             # Start the DAG 2 days ago from today
    'provide_context': True                 # Provide context to tasks, enabling access to task instance data
}


# Creating a new DAG for the drift analysis pipeline
dag = DAG(
    'Drift_Pipeline',                     # Unique identifier for the DAG
    default_args=default_args,            # Default arguments defined earlier for task configurations
    schedule_interval='0 0 * * 2',        # Schedule the DAG to run at midnight every Tuesday
    max_active_runs=1,                    # Limit to one active run of this DAG at a time
    tags=['ml_pipeline']                   # Tags for organizing and filtering DAGs in the Airflow UI
)

# Task to reset process flags in the database
op_reset_processes_flags = PythonOperator(
    task_id='reset_processes_flag',
    python_callable=get_flush_db_process_flags,
    op_kwargs={'db_path': db_path, 'drfit_db_name': drfit_db_name, 'flip': False},
    dag=dag,
    doc="""Reset process flags in the drift database to ensure a fresh start for the data pipeline.
          This task is executed at the beginning of each DAG run."""
)

# Task to create the drift database
op_create_db = PythonOperator(
    task_id='create_check_db', 
    python_callable=build_dbs,
    op_kwargs={'db_path': db_path, 'db_file_name': drfit_db_name},
    dag=dag,
    doc="""Create the drift database if it does not exist.
          This task sets up the necessary structure for storing drift-related data."""
)

# Task to create the feature store database
op_create_db_2 = PythonOperator(
    task_id='create_check_db_2', 
    python_callable=build_dbs,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
    dag=dag,
    doc="""Create the feature store database if it does not exist.
          This task establishes the database for storing feature engineering results."""
)

# Task to retrieve drift data
op_get_drift_data = PythonOperator(
    task_id='get_drift', 
    python_callable=get_drift,
    op_kwargs={
        'old_data_directory': old_data_directory,
        'new_data_directory': new_data_directory,
        'db_path': db_path,
        'drift_db_name': drift_db_name,
        'metric': metric,
        'start_date': start_date,
        'end_date': end_date,
        'chunk_size': 50000
    },
    dag=dag,
    doc="""Retrieve drift data by comparing old and new datasets.
          This task computes the drift metric over specified time periods and directories."""
)

# Task to load data from source
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
    doc="""Load data from specified directories into the database.
          This task facilitates the ingestion of new and old datasets for further processing."""
)

# Task to process membership data
op_process_members = PythonOperator(
    task_id='process_members', 
    python_callable=get_membership_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Process membership data from the feature store.
          This task retrieves and transforms membership data to prepare it for analysis."""
)

# Task to process transaction data
op_process_transactions = PythonOperator(
    task_id='process_transactions',
    python_callable=get_transaction_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Process transaction data from the feature store.
          This task retrieves and transforms transaction data to facilitate further analysis."""
)

# Task to process user log data
op_process_userlogs = PythonOperator(
    task_id='process_userlogs',
    python_callable=get_user_data_transform,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Process user log data from the feature store.
          This task retrieves and transforms user log data for analysis and modeling."""
)

# Task to merge processed data
op_merge = PythonOperator(
    task_id='merge_data',
    python_callable=get_final_data_merge,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Merge processed membership, transaction, and user log data into a unified dataset.
          This task ensures that all relevant data is consolidated for modeling."""
)

# Task to prepare data for modeling
op_process_data = PythonOperator(
    task_id='data_preparation', 
    python_callable=get_data_prepared_for_modeling,
    op_kwargs={
        'db_path': db_path,
        'db_file_name': db_file_name,
        'drfit_db_name': drfit_db_name,
        'date_columns': date_columns
    },
    dag=dag,
    doc="""Prepare the merged dataset for modeling.
          This task includes handling date columns and formatting the data as required by the modeling process."""
)

# Task for model training without hyperparameter tuning
op_model_training_without_tuning = PythonOperator(
    task_id='Model_Training_plain', 
    python_callable=get_train_model,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Train the model using the prepared dataset without hyperparameter tuning.
          This task performs standard model training using default parameters."""
)

# Task for model training with hyperparameter tuning
op_model_training_with_tuning = PythonOperator(
    task_id='Model_Training_hpTunning', 
    python_callable=get_train_model_hptune,
    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'drfit_db_name': drfit_db_name},
    dag=dag,
    doc="""Train the model using the prepared dataset with hyperparameter tuning.
          This task optimizes model performance by adjusting hyperparameters during training."""
)


# Email notification setup based on drift value
drift_cnx = sqlite3.connect(os.path.join(db_path, drfit_db_name))
try:
    drift = pd.read_sql('SELECT * FROM drift', drift_cnx)
    drift_value = drift.mean(axis=1)[0]
except Exception as e:
    drift_value = 0
    print(f"Error reading drift data: {e}")

# Prepare email content based on drift value
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if drift_value >= 0 and drift_value <= 10:
    send_email = EmailOperator(
        task_id='send_email', 
        to='charliethomasctg@gmail.com', 
        subject='Drift Pipeline Complete. Less Than 10% Drift', 
        html_content=f"We have detected less than 10 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}", 
        dag=dag,
        doc="""Send an email notification when drift is less than 10%.
              This task informs stakeholders about the minimal drift detected between datasets."""
    )
elif drift_value >= 10 and drift_value <= 20:
    send_email = EmailOperator(
        task_id='send_email', 
        to='charliethomasctg@gmail.com', 
        subject='Drift Pipeline Complete. Drift 10-20%', 
        html_content=f"We have detected 10-20 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}", 
        dag=dag,
        doc="""Send an email notification when drift is between 10% and 20%.
              This task updates stakeholders about the moderate drift detected between datasets."""
    )
elif drift_value >= 20 and drift_value <= 30:
    send_email = EmailOperator(
        task_id='send_email', 
        to='charliethomasctg@gmail.com', 
        subject='Drift Pipeline Complete. Drift 20-30%', 
        html_content=f"We have detected 20-30 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}", 
        dag=dag,
        doc="""Send an email notification when drift is between 20% and 30%.
              This task alerts stakeholders about the significant drift detected between datasets."""
    )
else:
    send_email = EmailOperator(
        task_id='send_email', 
        to='charliethomasctg@gmail.com', 
        subject='Drift Pipeline Complete. More than 30% Drift', 
        html_content=f"We have detected more than 30 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}. Please re-start the whole feature pre-processing, EDA and engineering processes again on Notebooks.", 
        dag=dag,
        doc="""Send an email notification when drift exceeds 30%.
              This task warns stakeholders to re-evaluate the feature engineering and modeling processes due to excessive drift detected."""
    )

# Define task dependencies
op_reset_processes_flags >> op_create_db >> op_create_db_2 >> op_get_drift_data >> op_load_data
op_load_data >> [op_process_members, op_process_userlogs, op_process_transactions]
op_process_members >> op_merge
op_process_userlogs >> op_merge
op_process_transactions >> op_merge
op_merge >> op_process_data >> op_model_training_without_tuning >> op_model_training_with_tuning >> send_email

