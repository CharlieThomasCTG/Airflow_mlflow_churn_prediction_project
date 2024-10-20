# function

import sys
# Building the DAG using the functions from data_process and model module
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
#from constants_model_building import *
import os 
import sqlite3
from sqlite3 import Error
import pandas as pd
import importlib.util
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from airflow.operators.email_operator import EmailOperator
from datetime import date
import mlflow
import mlflow.sklearn
import utils


# Path
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
mlflow_tracking_uri = "http://0.0.0.0:6006"
#ml_flow_path = "runs:/2b26c1f389b3485bb44b17cf85681cb6/models"
run_on = "old" #"old"
append=False
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'
mlflow_experiment_name = "Model_Building_Pipeline"
ml_flow_model_path = root_folder+ "/mlruns/1/336b8a558d9e452bb664f9b0dbc9ca39/artifacts/models/"
ml_flow_path = root_folder+ "/mlruns/1/336b8a558d9e452bb664f9b0dbc9ca39"
#Model
model_dump_path = "/airflow/models/"
metric='std'
short_exp_name_identifier = "without_dateprep"

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

#utils = module_from_file("utils", "home/charliethomasctg/airflow/scripts/utils.py")


#make sure to run mlflow server before this. 
experiment_name = mlflow_experiment_name+'_'+date.today().strftime("%d_%m_%Y")+'_'+short_exp_name_identifier
mlflow.set_tracking_uri(mlflow_tracking_uri)

try:
    # Creating an experiment
    logging.info("Creating mlflow experiment")
    mlflow.create_experiment(experiment_name)
except:
    pass
# Setting the environment with the created experiment
mlflow.set_experiment(experiment_name)


# Declare Default arguments for the DAG
default_args = {
    'owner': 'CharlieThomas',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'provide_context': True
}


# creating a new dag
dag = DAG('Model_Building_Pipeline', default_args=default_args, schedule_interval='0 0 * * 2', max_active_runs=1,tags=['ml_pipeline'])

# Integrating different operatortasks in airflow dag

op_reset_processes_flags = PythonOperator(task_id='reset_processes_flag',
                                         python_callable=utils.get_flush_db_process_flags,
                                         op_kwargs={'db_path': db_path,'drfit_db_name':drfit_db_name},
                                         dag=dag)


op_create_db = PythonOperator(task_id='create_check_db', 
                            python_callable=utils.build_dbs,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
                            dag=dag)


op_load_data = PythonOperator(task_id='load_data', 
                                python_callable=utils.load_data_from_source,
                                  op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name,
                                             'old_data_directory':old_data_directory,
                                             'new_data_directory':new_data_directory,
                                            'run_on':run_on,
                                              'start_date':start_date,
                                             'end_date':end_date},
                              dag=dag)
    

op_process_transactions = PythonOperator(task_id='process_transactions',
                                         python_callable=utils.get_membership_data_transform,
                                         op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                         dag=dag)

op_process_members = PythonOperator(task_id='process_members', 
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

op_model_training_with_tuning = PythonOperator(task_id='Model_Training_hpTunning', 
                            python_callable=utils.get_train_model_hptune,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,'drfit_db_name':drfit_db_name},
                            dag=dag)


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email = EmailOperator( task_id='send_email', 
                                to='charliethomasctg@gmail.com', 
                                subject='Model Building Pipeline Execution Finished', 
                                html_content=f"The execution of Model Building pipeline is finished @ {timestamp}. Check Airflow Logs for more details or check MLFLOW for Final Model", 
                                dag=dag)




# Set the task sequence
op_reset_processes_flags.set_downstream(op_create_db)
op_create_db.set_downstream(op_load_data)
op_load_data.set_downstream([op_process_members,op_process_userlogs,op_process_transactions])
op_process_members.set_downstream(op_merge)
op_process_userlogs.set_downstream(op_merge)
op_process_transactions.set_downstream(op_merge)
op_merge.set_downstream(op_process_data)
op_process_data.set_downstream(op_model_training_with_tuning)
op_model_training_with_tuning.set_downstream(send_email)

