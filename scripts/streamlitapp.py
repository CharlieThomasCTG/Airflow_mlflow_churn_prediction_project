import streamlit as st
import pandas as pd
import sqlite3
import mlflow
import utils
import os

# Setting up all directories
root_folder = "/home/charliethomasctg/airflow"
database_path = os.path.join(root_folder, "database/") 
data_profile_path = os.path.join(root_folder, "data/profile_report")
mlflow_tracking_uri = "http://0.0.0.0:6006"
ml_flow_path = root_folder + "/mlruns/1/336b8a558d9e452bb664f9b0dbc9ca39"
mlflow_model_path = root_folder + "/mlruns/1/336b8a558d9e452bb664f9b0dbc9ca39/artifacts/models"

# Database configuration
db_path = root_folder + "/database/"
db_file_name = "feature_store_v01.db"
drift_db_name = "drift_db_name.db"

# Additional configuration for viewing final predictions
final_predictions_db_name = "db_final_name.db"

# Sidebar Configuration
st.sidebar.header("Machine Learning Management App")
option = st.sidebar.selectbox("Choose an option", ["Make Prediction", "View Predictions"])

# Streamlit app configuration inputs
db_path = st.sidebar.text_input("Database Path", value=db_path)
mlflow_model_path = st.sidebar.text_input("MLFlow Model Path", value=mlflow_model_path)



# Function to make predictions
def make_prediction():
    st.header("Make Predictions")
    st.write("Upload a new dataset for predictions.")

    # Add a unique key to the file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader_1")
    
    if uploaded_file is not None:
        # Read the uploaded file
        new_data = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.dataframe(new_data.head())

        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    # Ensure all necessary arguments are passed
                    result = utils.get_predict(
                        db_path,
                        db_file_name,
                        mlflow_model_path,
                        drift_db_name
                    )

                    # Print the result to see what it looks like
                    print("Result from prediction:", result)

                    # Convert to DataFrame if not already one
                    if not isinstance(result, pd.DataFrame):
                        # Attempt to convert to DataFrame
                        try:
                            result = pd.DataFrame(result)
                        except Exception as e:
                            st.error(f"Error converting to DataFrame: {e}")
                            return

                    # Display the prediction results
                    st.success("Prediction completed successfully!")
                    st.write("Prediction Results:")
                    st.dataframe(result)  # Display the prediction results

                except Exception as e:
                    st.error(f"Error during prediction: {e}")


# Function to view predictions
def view_final_predictions(db_path, db_file_name):
    st.header("View Final Predictions")
    
    try:
        # Connect to the SQLite database
        cnx = sqlite3.connect(db_path + db_file_name)

        # Query to retrieve data from the predictions table
        query = 'SELECT * FROM predictions'  # Make sure this matches your table name
        
        # Execute the query and return the result as a DataFrame
        final_predictions_df = pd.read_sql(query, cnx)

        # Close the connection
        cnx.close()

        # Display the predictions
        if final_predictions_df.empty:
            st.warning("No predictions found in the database.")
        else:
            st.write("Final Predictions:")
            st.dataframe(final_predictions_df)

    except Exception as e:
        st.error(f"Error retrieving Final Predictions table: {e}")


# Streamlit App Main Section
if option == "Make Prediction":
    make_prediction()
elif option == "View Predictions":
    view_final_predictions(db_path, db_file_name)  # Call the view predictions function
