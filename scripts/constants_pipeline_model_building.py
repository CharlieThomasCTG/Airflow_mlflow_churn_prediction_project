import os
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