import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from pycaret.classification import *
import mlflow

# Setting up all directories
root_folder = "/home/charliethomasctg/airflow"
database_path = root_folder + "/database/"

# Create SQLite connections using utility function
utils.create_sqlit_connection(database_path, "mlflow_v01.db")
utils.create_sqlit_connection(database_path, "feature_store_v01.db")
utils.create_sqlit_connection(database_path, "drift_db_name.db")
utils.create_sqlit_connection(database_path, "drfit_db_name.db")

# Define the virtual environment path
venv_path = "/home/charliethomasctg/venv/bin/activate"  # Update with the correct venv path

# Function to start Airflow services and MLflow server
def start_services(venv_path):
    try:
        # Start the Airflow webserver
        subprocess.Popen(
            f"bash -c 'source {venv_path} && airflow webserver --port 8080'",
            shell=True
        )
        print("Airflow webserver started on port 8080.")
        
        # Start the Airflow scheduler
        subprocess.Popen(
            f"bash -c 'source {venv_path} && airflow scheduler'",
            shell=True
        )
        print("Airflow scheduler started.")

        # Start the MLflow server
        subprocess.Popen(
            f"bash -c 'source {venv_path} && mlflow server "
            "--backend-store-uri sqlite:////home/charliethomasctg/airflow/database/mlflow_v01.db "
            "--default-artifact-root /home/charliethomasctg/airflow/mlruns "
            "--port 6006 --host 0.0.0.0'",
            shell=True
        )
        print("MLflow server started on port 6006.")
        
        # Wait a few seconds to allow Airflow and MLflow to start properly
        time.sleep(10)
        
        # Start the Streamlit app
        subprocess.Popen(
            f"bash -c 'source {venv_path} && streamlit run /home/charliethomasctg/airflow/scripts/streamlitapp.py "
            "--server.port 8500 --server.address 0.0.0.0'",
            shell=True
        )
        print("Streamlit app started on port 8500.")
        
    except Exception as e:
        print(f"Failed to start services: {e}")

# Start services
start_services(venv_path)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:6006")
