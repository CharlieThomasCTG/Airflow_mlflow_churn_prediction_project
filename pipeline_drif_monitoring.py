#pip install scikit-optimize
# Building the DAG using the functions from data_process and model module
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
#from constants_drift import *
import os 
import sqlite3
from sqlite3 import Error
import pandas as pd
import importlib.util
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from datetime import date
import mlflow
import mlflow.sklearn
import utils

import os
import sys

new_directory = r'\\wsl.localhost\Ubuntu\home\charliethomasctg\airflow'
current_directory = os.getcwd()
scripts_path = os.path.abspath(os.path.join(os.getcwd(), '../scripts'))


# function
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
mlflow_tracking_uri = "http://0.0.0.0:6007"
ml_flow_path = "runs:/2/cb66e22bcbf74ded99dc219eb29e7609/"
run_on = "old" #"old"
append=False
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'
mlflow_experiment_name = "Model_Building_Pipeline_Drift"

#Model
model_dump_path = "/models/"
metric='std'
short_exp_name_identifier = "without_dateprep"

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/home/charliethomasctg/airflow/scripts/utils.py")

#MLFlow
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
dag = DAG('Drift_Pipeline', default_args=default_args, schedule_interval='0 0 * * 2', max_active_runs=1,tags=['ml_pipeline'])

op_reset_processes_flags = PythonOperator(task_id='reset_processes_flag',
                                         python_callable=utils.get_flush_db_process_flags,
                                         op_kwargs={'db_path': db_path,'drfit_db_name':drfit_db_name,
                                                   'flip':False},
                                         dag=dag)


op_create_db = PythonOperator(task_id='create_check_db', 
                            python_callable=utils.build_dbs,
                            op_kwargs={'db_path': db_path, 'db_file_name': drfit_db_name},
                            dag=dag)

op_create_db_2 = PythonOperator(task_id='create_check_db_2', 
                            python_callable=utils.build_dbs,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
                            dag=dag)


op_get_drift_data = PythonOperator(task_id='get_drift', 
                            python_callable=utils.get_drift,
                            op_kwargs={'old_data_directory':old_data_directory,
                                             'new_data_directory':new_data_directory,
                                       'db_path': db_path,
                                       'drfit_db_name':drfit_db_name,
                                       'metric':metric,
                                       'start_date':start_date,
                                             'end_date':end_date,
                                       },
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
                                       'date_columns':date_columns
                                      },
                            dag=dag)


op_model_training_without_tuning = PythonOperator(task_id='Model_Training_plain', 
                            python_callable=utils.get_train_model,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,'drfit_db_name':drfit_db_name},
                            dag=dag)


op_model_training_with_tuning = PythonOperator(task_id='Model_Training_hpTunning', 
                            python_callable=utils.get_train_model_hptune,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,'drfit_db_name':drfit_db_name},
                            dag=dag)

# op_predict_data = PythonOperator(task_id='Prediction', 
#                             python_callable=utils.get_predict,
#                             op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,'ml_flow_path':ml_flow_path,'drfit_db_name':drfit_db_name},
#                             dag=dag)


# Email Triggers 

drift_cnx = sqlite3.connect(db_path+drfit_db_name)
try:
    drift = pd.read_sql('select * from drift', drift_cnx)
    drift_value = drift.mean(axis=1)[0]
except:
    drift_value = 0

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if drift_value >= 0 and drift_value <=10:
    send_email = EmailOperator( task_id='send_email', 
                                to='charliethomasctg@gmail.com', 
                                subject='Drift Pipeline Complete. Less Than 10% Drift', 
                                html_content=f"We have detected less than 10 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}", 
                                dag=dag)
elif drift_value >= 10 and drift_value <=20:
    send_email = EmailOperator( task_id='send_email', 
                                to='charliethomasctg@gmail.com', 
                                subject='Drift Pipeline Complete. Drift 10-20% Drift', 
                                html_content=f"We have detected 10-20 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}", 
                                dag=dag)
elif drift_value >= 20 and drift_value <=30:
    send_email = EmailOperator( task_id='send_email', 
                                to='charliethomasctg@gmail.com', 
                                subject='Drift Pipeline Complete. Drift 10-20% Drift', 
                                html_content=f"We have detected 20-30 percent (metric averaged) drift between new and old User Logs and Transaction Data @ {timestamp}", 
                                dag=dag)
else:
    send_email = EmailOperator( task_id='send_email', 
                                to='charliethomasctg@gmail.com', 
                                subject='Drift Pipeline Complete. More than 30% Drift', 
                                html_content=f"We have detected more than 30 percent (metric averaged) drift btween new and old User Logs and Transaction Data @ {timestamp}. Please re-start the whole featuer pre-processing, EDA and engineering processes again on Notebooks.", 
                                dag=dag)

op_reset_processes_flags.set_downstream(op_create_db)
op_create_db.set_downstream(op_create_db_2)
op_create_db_2.set_downstream(op_get_drift_data)
op_get_drift_data.set_downstream(op_load_data)
op_load_data.set_downstream([op_process_members,op_process_userlogs,op_process_transactions])
op_process_members.set_downstream(op_merge)
op_process_userlogs.set_downstream(op_merge)
op_process_transactions.set_downstream(op_merge)

op_merge.set_downstream(op_process_data)
op_process_data.set_downstream(op_model_training_without_tuning)
op_model_training_without_tuning.set_downstream(op_model_training_with_tuning)
op_model_training_with_tuning.set_downstream(send_email)
