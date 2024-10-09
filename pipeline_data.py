# function
import os
import sys

new_directory = r'\\wsl.localhost\Ubuntu\home\charliethomasctg\airflow'
current_directory = os.getcwd()
scripts_path = os.path.abspath(os.path.join(os.getcwd(), '../scripts'))

def change_directory(current_directory, new_directory,scripts_path):
    # Get the current working directory
    print(f'Current directory: {current_directory}')
    # Define the path to change to
    new_directory = r'\\wsl.localhost\Ubuntu\home\charliethomasctg\airflow'
    try:
        # Change the current working directory
        os.chdir(new_directory)
        # Verify the change
        current_directory = os.getcwd()
        print(f'Current directory changed to: {current_directory}')
    except FileNotFoundError:
        print(f'Error: The directory "{new_directory}" does not exist.')
    except PermissionError:
        print(f'Error: Permission denied to change to "{new_directory}".')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    # Add the scripts directory to the Python path
    sys.path.append(scripts_path)

    
change_directory(current_directory, new_directory, scripts_path)



# Building the DAG using the functions from data_process and model module
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
#from constants_data_pipeline import *
import os 
import sqlite3
from sqlite3 import Error
import pandas as pd
import importlib.util
from datetime import datetime
import utils



# Setting up all directory
root_folder = "/home/charliethomasctg/airflow"
database_path = root_folder+"/database/"
data_directory = root_folder+"/data/raw/"
data_profile_path = root_folder+"/data/profile_report/"
intermediate_data_path = root_folder+"/data/interim/"
final_processed_data_path = root_folder+"/data/processed/"

old_data_directory = root_folder+"/data/raw/"
new_data_directory = root_folder+"/data/new/"
intermediate_path = root_folder+"/data/interim/"


# Database
db_path = root_folder+"/database/"
db_file_name = "feature_store_v01.db"
drfit_db_name = "drift_db_name.db"
date_columns = ['registration_init_time','transaction_date_min','transaction_date_max','membership_expire_date_max','last_login']


# Mlflow
ml_flow_model_path = "runs:/c8b4e75b247047fbbeebcb2be6b073ca/models"
run_on = "old"
append=True
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'



def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/home/charliethomasctg/airflow/scripts/utils.py")

# Declare Default arguments for the DAG
default_args = {
    'owner': 'CharlieThomas',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'provide_context': True,
}


# creating a new dag
dag = DAG('Data_End2End_Processing', default_args=default_args, schedule_interval='0 0 * * 2', max_active_runs=1,tags=['data_pipeline'])

# Integrating different operatortasks in airflow dag

op_reset_processes_flags = PythonOperator(task_id='reset_processes_flag',
                                         python_callable=utils.get_flush_db_process_flags,
                                         op_kwargs={'db_path': db_path,'drfit_db_name':drfit_db_name},
                                         dag=dag)


op_load_data = PythonOperator(task_id='load_data', 
                                python_callable=utils.load_data_from_source,
                                  op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name,
                                             'old_data_directory':old_data_directory,
                                             'new_data_directory':new_data_directory,
                                            'run_on':run_on,
                                              'start_date':start_date,
                                             'end_date':end_date,
                                           },
                              dag=dag)

op_process_members = PythonOperator(task_id='process_members', 
                                    python_callable=utils.get_membership_data_transform,
                                    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                    dag=dag)

op_process_transactions = PythonOperator(task_id='process_transactions',
                                         python_callable=utils.get_transaction_data_transform,
                                         op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                         dag=dag)


op_process_userlogs = PythonOperator(task_id='process_userlogs',
                                    python_callable=utils.get_user_data_transform,
                                    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                    dag=dag)

op_merge = PythonOperator(task_id='merge_data',
                        python_callable=utils.get_final_data_merge,
                        op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                        dag=dag)


op_process_data = PythonOperator(task_id='data_preparation', 
                            python_callable=utils.get_data_prepared_for_modeling,
                            op_kwargs={'db_path': db_path,
                                       'db_file_name': db_file_name,
                                       'drfit_db_name':drfit_db_name,
                                       'date_columns':date_columns,
                                       'date_transformation':date_transformation
                                      },
                            dag=dag)


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email = EmailOperator( task_id='send_email', 
                                to='charliethomasctg@gmail.com', 
                                subject='Data Pipeline Execution Finished', 
                                html_content=f"The execution of data pipeline is finished @ {timestamp}. Check Airflow Logs for more details or check SQLite Backend for Transformed Data", 
                                dag=dag)


# Set the task sequence
op_reset_processes_flags.set_downstream(op_load_data)
op_load_data.set_downstream([op_process_members,op_process_userlogs,op_process_transactions])
op_process_members.set_downstream(op_merge)
op_process_userlogs.set_downstream(op_merge)
op_process_transactions.set_downstream(op_merge)
op_merge.set_downstream(op_process_data)
op_process_data.set_downstream(send_email)