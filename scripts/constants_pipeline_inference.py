import os
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