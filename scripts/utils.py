import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from ydata_profiling import ProfileReport
import sqlite3
from sqlite3 import Error
from pycaret.classification import *
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV # run pip install scikit-optimize
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from datetime import datetime
from datetime import date
import logging
import requests
from typing import Optional,Dict

###############################################################################################################################################################################

import pandas as pd

def load_data(file_path_list):
    """
    Loads data from a list of file paths and returns it as a list of DataFrames.
    
    This function reads CSV files from the specified list of file paths. It attempts to load each file,
    catching exceptions if any issues occur (such as file not found or data read errors). If an error
    occurs for a particular file, it logs the error and continues with the next file.
    
    Parameters:
    ----------
    file_path_list : list of str
        A list containing the file paths to the CSV files to be loaded.
    
    Returns:
    -------
    list of pd.DataFrame
        A list of pandas DataFrames, where each DataFrame corresponds to a successfully loaded CSV file.
    
    """
    data = []
    for eachfile in file_path_list:
        try:
            df = pd.read_csv(eachfile)
            data.append(df)
        except FileNotFoundError:
            print(f"File not found: {eachfile}")
        except pd.errors.EmptyDataError:
            print(f"Empty file or no data: {eachfile}")
        except pd.errors.ParserError:
            print(f"Parsing error in file: {eachfile}")
        except Exception as e:
            print(f"An unexpected error occurred while reading {eachfile}: {e}")
    return data

###############################################################################################################################################################################
def remove_outliers(df):
    """
    Removes outliers from a DataFrame based on the Interquartile Range (IQR) method.
    
    This function iterates over each numeric column in the DataFrame, calculating the 
    first (Q1) and third (Q3) quartiles to determine the IQR. It then defines lower 
    and upper limits as 1.5 * IQR below Q1 and above Q3, respectively, and removes rows 
    with values outside this range.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame from which outliers are to be removed.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with outliers removed based on the IQR method for each numeric column.

    """
    # Start with the original DataFrame
    cleaned_df = df.copy()
    
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype in ['int', 'float']:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define lower and upper limits
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Remove outliers
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_limit) & 
                                    (cleaned_df[col] <= upper_limit)]

    return cleaned_df


###############################################################################################################################################################################
def convert_float_columns(df):
    """
    Converts all columns with data type float16 to float32 in a DataFrame.
    
    This function identifies columns in the DataFrame that are of type float16 
    and converts them to float32 to increase precision and compatibility with 
    libraries that may not support float16. The function modifies the DataFrame 
    in place and returns it with the updated data types.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame with potential float16 columns.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with all float16 columns converted to float32.
    
    """
    # Convert all float16 columns to float32
    for col in df.select_dtypes(include=['float16']).columns:
        df[col] = df[col].astype('float32')
    return df


###############################################################################################################################################################################
def compress_dataframes(list_of_dfs):
    """
    Compresses the memory usage of a list of DataFrames by downcasting numeric columns.
    
    This function takes a list of DataFrames and attempts to reduce their memory footprint 
    by converting integer columns to the smallest possible integer type (int8, int16, int32, 
    or int64) and float columns to float16. It calculates the original and compressed sizes 
    of each DataFrame and returns this information alongside the modified DataFrames.

    Parameters:
    ----------
    list_of_dfs : list of pd.DataFrame
        A list containing DataFrames to be compressed.

    Returns:
    -------
    list of tuple
        A list of tuples, where each tuple contains the compressed DataFrame, 
        its original memory size (in MB), and its compressed memory size (in MB).
    
    """
    final_df = []
    for eachdf in list_of_dfs:
        original_size = (eachdf.memory_usage(index=True).sum()) / 1024**2
        int_cols = list(eachdf.select_dtypes(include=['int']).columns)
        float_cols = list(eachdf.select_dtypes(include=['float']).columns)
        
        # Downcasting integer columns
        for col in int_cols:
            if ((np.max(eachdf[col]) <= 127) and (np.min(eachdf[col]) >= -128)):
                eachdf[col] = eachdf[col].astype(np.int8)
            elif ((np.max(eachdf[col]) <= 32767) and (np.min(eachdf[col]) >= -32768)):
                eachdf[col] = eachdf[col].astype(np.int16)
            elif ((np.max(eachdf[col]) <= 2147483647) and (np.min(eachdf[col]) >= -2147483648)):
                eachdf[col] = eachdf[col].astype(np.int32)
            else:
                eachdf[col] = eachdf[col].astype(np.int64)
        
        # Downcasting float columns
        for col in float_cols:
            eachdf[col] = eachdf[col].astype(np.float16)
        
        compressed_size = (eachdf.memory_usage(index=True).sum()) / 1024**2
        
        final_df.append((eachdf, original_size, compressed_size))
        
    return final_df


###############################################################################################################################################################################
def count_plot(dataframe, list_of_columns):
    """
    Generate and display count plots for specified columns in a DataFrame.

    This function takes a DataFrame and a list of column names, and it generates a count plot 
    for each specified column. If the column's data type is integer, the unique values are 
    sorted before plotting. The function utilizes Seaborn and Matplotlib for visualization.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data for plotting.

    list_of_columns : list of str
        A list of column names for which to generate count plots.

    Returns:
    -------
    None

    Raises:
    ------
    ValueError: If any of the specified columns do not exist in the DataFrame.
    TypeError: If the data type of any specified column is not suitable for plotting.

    """
    try:
        for eachcol in list_of_columns:
            plt.figure(figsize=(15, 5))

            # Check if the column exists in the DataFrame
            if eachcol not in dataframe.columns:
                raise ValueError(f"Column '{eachcol}' does not exist in the DataFrame.")

            unique_features = dataframe[eachcol].unique()

            # Check if the column data type is suitable for plotting
            if not pd.api.types.is_numeric_dtype(dataframe[eachcol]) and not pd.api.types.is_object_dtype(dataframe[eachcol]):
                raise TypeError(f"Column '{eachcol}' has an unsupported data type for plotting.")

            if dataframe[eachcol].dtype == 'int64':
                unique_features = sorted(unique_features)

            sns.countplot(x=eachcol, data=dataframe, order=unique_features)
            plt.xlabel(eachcol)
            plt.ylabel('Count')
            plt.title("Frequency plot of {} Count".format(eachcol))
            plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")


###############################################################################################################################################################################
def fix_time_in_df(dataframe, column_name, expand=False):
    """
    Convert a specified column in a DataFrame to datetime format and optionally 
    expand it into separate year, month, and day components.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column to be converted.

    column_name : str
        The name of the column in the DataFrame to be converted to datetime.

    expand : bool, optional
        If True, create additional columns for year, month, and day. Default is False.

    Returns:
    -------
    pd.Series or pd.DataFrame
        If `expand` is False, returns a Series of datetime values.
        If `expand` is True, returns a new DataFrame with additional date components.

    Raises:
    ------
    ValueError: If the specified column does not exist in the DataFrame or cannot be converted to datetime.
    """
    try:
        # Check if the column exists in the DataFrame
        if column_name not in dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        if not expand:
            dataframe[column_name] = dataframe[column_name].astype('str')
            return pd.to_datetime(dataframe[column_name])
        else:
            dataframe_new = dataframe.copy()
            dataframe_new[column_name] = dataframe_new[column_name].astype('str')
            dataframe_new[column_name] = pd.to_datetime(dataframe_new[column_name])

            # Extracting the datetime components
            dataframe_new[f"{column_name}_year"] = pd.DatetimeIndex(dataframe_new[column_name]).year
            dataframe_new[f"{column_name}_month"] = pd.DatetimeIndex(dataframe_new[column_name]).month
            dataframe_new[f"{column_name}_day"] = pd.DatetimeIndex(dataframe_new[column_name]).day_name()

            return dataframe_new
    except Exception as e:
        print(f"An error occurred: {e}")

    
###############################################################################################################################################################################

def get_data_profile(dataframe, html_save_path, 
                     embed_in_cell=True, take_sample=False, sample_frac=0.5, dataframe_name="data"):
    """
    Generate a data profile report for a given DataFrame using the Pandas Profiling library.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be profiled.

    html_save_path : str
        The directory path where the HTML report will be saved if not embedding in a notebook.

    embed_in_cell : bool, optional
        If True, returns the profile report to be embedded in a Jupyter Notebook cell. 
        If False, saves the report as an HTML file. Default is True.

    take_sample : bool, optional
        If True, takes a random sample of the DataFrame. Default is False.

    sample_frac : float, optional
        The fraction of the DataFrame to sample if `take_sample` is True. Default is 0.5.

    dataframe_name : str, optional
        The name to be used in the title of the report and the saved file. Default is "data".

    Returns:
    -------
    If `embed_in_cell` is True, returns the profile report as an iframe.
    If `embed_in_cell` is False, returns a message with the path to the saved HTML report.

    Raises:
    ------
    ValueError: If the `html_save_path` does not exist or is not a directory.
    
    """
    try:
        # Check if the save path is valid
        if not os.path.isdir(html_save_path):
            raise ValueError(f"The specified path '{html_save_path}' is not a valid directory.")

        if take_sample:
            dataframe = dataframe.sample(frac=sample_frac)

        profile = ProfileReport(dataframe, title=f"{dataframe_name} Data Summary Report")

        if embed_in_cell:
            return profile.to_notebook_iframe()
        else:
            timestamp = str(int(time.time()))
            filename = f"{dataframe_name}_data_profile_{timestamp}"
            profile.to_file(os.path.join(html_save_path, filename + ".html"))
            return "Your Data Profile has been saved at:", os.path.join(html_save_path, filename + ".html")
            
    except Exception as e:
        print(f"An error occurred: {e}")
 
     
###############################################################################################################################################################################   
def get_data_describe(dataframe, round_num=2):
    """
    Generate descriptive statistics for a given DataFrame.

    This function computes summary statistics of the DataFrame, rounding 
    the results to the specified number of decimal places.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame for which descriptive statistics are to be calculated.

    round_num : int, optional
        The number of decimal places to round the results to. Default is 2.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the descriptive statistics of the input DataFrame.

    Raises:
    ------
    ValueError: If the input is not a valid DataFrame.
    """
    try:
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The input must be a valid pandas DataFrame.")
        
        return round(dataframe.describe(), round_num)
    
    except Exception as e:
        print(f"An error occurred: {e}")


###############################################################################################################################################################################
def get_data_na_values(dataframe, round_num=2):
    """
    Calculate the percentage of missing values in each column of a DataFrame.

    This function computes the percentage of missing values for each column
    in the given DataFrame, rounding the results to the specified number of
    decimal places.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame for which the percentage of missing values is to be calculated.

    round_num : int, optional
        The number of decimal places to round the results to. Default is 2.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the percentage of missing values for each column in the input DataFrame.

    Raises:
    ------
    ValueError: If the input is not a valid DataFrame.

    """
    try:
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The input must be a valid pandas DataFrame.")
        
        # Calculate percentage of missing values
        return pd.DataFrame({'%missing_values': round(dataframe.isna().sum() / dataframe.shape[0], round_num)})
    
    except Exception as e:
        print(f"An error occurred: {e}")


###############################################################################################################################################################################
def get_fill_na_dataframe(dataframe, column_name, value='mean'):
    """
    Fill missing values in a specified column of a DataFrame.

    This function replaces NaN (missing) values in the specified column of the
    DataFrame with a specified value. The value can be the mean, mode, or
    a custom value provided by the user.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame in which missing values will be filled.

    column_name : str
        The name of the column in which to fill missing values.

    value : {'mean', 'mode', float, str}, optional
        The value to fill missing entries. It can be 'mean' to use the mean of the column,
        'mode' to use the mode of the column, or a specific value (float or string).
        Default is 'mean'.

    Returns:
    -------
    pd.Series
        A Series with the missing values filled.

    Raises:
    ------
    ValueError: If the column_name does not exist in the DataFrame.
    TypeError: If the provided value type is incorrect for the column.
    """
    try:
        if column_name not in dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        # Fill missing values based on the specified value
        if value != 'mean' and value != 'mode':
            return dataframe[column_name].fillna(value)
        elif value == 'mean':
            mean_value = dataframe[column_name].mean()
            return dataframe[column_name].fillna(mean_value)
        elif value == 'mode':
            mode_value = dataframe[column_name].mode()[0]  # Take the first mode
            return dataframe[column_name].fillna(mode_value)

    except TypeError:
        print(f"Invalid value type provided for filling NaN in column '{column_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    
###############################################################################################################################################################################
def get_convert_column_dtype(dataframe, column_name, data_type='str'):
    """
    Convert the data type of a specified column in a DataFrame.

    This function changes the data type of a specified column to the desired 
    data type. Supported data types include string, integer, and float.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column to convert.

    column_name : str
        The name of the column whose data type will be changed.

    data_type : {'str', 'int', 'float'}, optional
        The target data type to convert the column to. Default is 'str'.

    Returns:
    -------
    pd.Series
        The column with the converted data type.

    Raises:
    ------
    ValueError: If the column_name does not exist in the DataFrame.
    TypeError: If the data type conversion fails due to incompatible data.

    """
    try:
        if column_name not in dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        if data_type == 'str':
            return dataframe[column_name].astype('str')
        elif data_type == 'int':
            return dataframe[column_name].astype('int')
        elif data_type == 'float':
            return dataframe[column_name].astype('float')
        else:
            raise ValueError(f"Unsupported data type '{data_type}' specified. Use 'str', 'int', or 'float'.")

    except ValueError as ve:
        print(ve)
    except TypeError as te:
        print(f"TypeError: Could not convert column '{column_name}' to {data_type}. Please check the data.")
    except Exception as e:
        print(f"An error occurred: {e}")

    
###############################################################################################################################################################################    
def get_groupby(dataframe, by_column, agg_dict=None, agg_func='mean', simple_agg_flag=True, reset_index=True):
    """
    Perform a group-by operation on a DataFrame with specified aggregations.

    This function allows for grouping a DataFrame by a specified column and applying 
    aggregation functions to the grouped data. It supports both simple aggregation 
    (using a single aggregation function) and complex aggregation (using a dictionary 
    of aggregation functions).

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to perform the group-by operation on.

    by_column : str
        The name of the column to group by.

    agg_dict : dict, optional
        A dictionary specifying the aggregation functions for specific columns, 
        used if `simple_agg_flag` is set to False. Default is None.

    agg_func : str, optional
        A string representing the aggregation function to apply if `simple_agg_flag` 
        is set to True. Default is 'mean'.

    simple_agg_flag : bool, optional
        If True, uses a single aggregation function; if False, uses `agg_dict`. 
        Default is True.

    reset_index : bool, optional
        If True, resets the index of the resulting DataFrame. Default is True.

    Returns:
    -------
    pd.DataFrame
        The grouped DataFrame with the applied aggregations.

    Raises:
    ------
    ValueError: If the `by_column` does not exist in the DataFrame or if both 
                `agg_func` and `agg_dict` are None.

    """
    try:
        if by_column not in dataframe.columns:
            raise ValueError(f"Column '{by_column}' does not exist in the DataFrame.")
        
        if simple_agg_flag:
            if agg_func is None:
                raise ValueError("If using simple aggregation, `agg_func` cannot be None.")
            return dataframe.groupby(by_column).agg(agg_func).reset_index() if reset_index else dataframe.groupby(by_column).agg(agg_func)
        else:
            if agg_dict is None:
                raise ValueError("If using complex aggregation, `agg_dict` cannot be None.")
            return dataframe.groupby(by_column).agg(agg_dict).reset_index() if reset_index else dataframe.groupby(by_column).agg(agg_dict)

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

        
###############################################################################################################################################################################       
def get_merge(dataframe1, dataframe2, on, axis=1, how='inner'):
    """
    Merge two DataFrames on a specified column or index with specified join type.

    This function performs a merge operation between two DataFrames, allowing control over
    the join type, the column(s) to join on, and the axis to merge along.

    Parameters:
    ----------
    dataframe1 : pd.DataFrame
        The first DataFrame to merge.
        
    dataframe2 : pd.DataFrame
        The second DataFrame to merge.

    on : str or list
        The column or index level name(s) to join on.

    axis : int, default 1
        Axis to merge along; 1 represents column-wise joining.

    how : str, default 'inner'
        Type of join to perform. Options are 'inner', 'outer', 'left', and 'right'.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the merged result of the two input DataFrames.

    Raises:
    ------
    ValueError:
        If any specified column(s) in 'on' are missing in either DataFrame.
    TypeError:
        If the inputs are not of type pd.DataFrame.
    Exception:
        For any other errors encountered during the merge process.
    """
    try:
        # Ensure both inputs are DataFrames
        if not isinstance(dataframe1, pd.DataFrame) or not isinstance(dataframe2, pd.DataFrame):
            raise TypeError("Both inputs must be pandas DataFrames.")
        
        # Perform the merge operation
        return dataframe1.merge(dataframe2, on=on, how=how)

    except KeyError as ke:
        print(f"Column specified in 'on' not found in one of the DataFrames: {ke}")
    except TypeError as te:
        print(te)
    except Exception as e:
        print(f"An error occurred during merge: {e}")

###############################################################################################################################################################################
def get_fix_skew_with_log(dataframe, columns, replace_inf=True, replace_inf_with=0):
    """
    Apply a logarithmic transformation to specified columns of a DataFrame to reduce skewness.

    This function computes the natural logarithm of the specified columns in the given DataFrame.
    If the logarithm of any value results in infinity (due to logarithm of zero or negative numbers),
    the function can replace these infinite values with a specified value.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data to transform.

    columns : list
        A list of column names to apply the logarithmic transformation.

    replace_inf : bool, default True
        If True, replaces infinite values resulting from the logarithm with `replace_inf_with`.

    replace_inf_with : numeric, default 0
        The value to replace infinite values with, if `replace_inf` is True.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the logarithmically transformed columns and the other columns unchanged.

    Raises:
    ------
    ValueError: If any of the specified columns are not present in the DataFrame.
    
    """
    try:
        # Check if all specified columns are in the DataFrame
        for col in columns:
            if col not in dataframe.columns:
                raise ValueError(f"Column '{col}' is not present in the DataFrame.")

        # Apply logarithmic transformation
        if replace_inf:
            dataframe_log = np.log(dataframe[columns]).replace([np.inf, -np.inf], replace_inf_with)
        else:
            dataframe_log = np.log(dataframe[columns])

        # Combine the transformed columns with the rest of the DataFrame
        return pd.concat([dataframe_log, dataframe.drop(columns, axis=1)], axis=1)

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")


    
###############################################################################################################################################################################        
def get_save_intermediate_data(dataframe, path, filename="data_interim"):
    """
    Save the given DataFrame to a CSV file at the specified path with a timestamp.

    This function saves the provided DataFrame as a CSV file. The filename includes a timestamp
    to ensure uniqueness. If a filename is provided, it will be appended with a timestamp before
    saving.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be saved.

    path : str
        The directory path where the CSV file will be saved.

    filename : str, default "data_interim"
        The base filename (without extension) to use for the saved file.

    Returns:
    -------
    str
        A message indicating the location of the saved CSV file.

    Raises:
    ------
    ValueError: If the provided path is not a directory or is invalid.
    IOError: If there is an error writing the file.
    """
    try:
        # Check if the provided path is a valid directory
        if not os.path.isdir(path):
            raise ValueError(f"The provided path '{path}' is not a valid directory.")

        # Generate the full filename with timestamp
        filename = filename + "_" + str(int(time.time())) + ".csv"
        # Save the DataFrame to CSV
        dataframe.to_csv(os.path.join(path, filename), index=False)

        return "Data Saved Here :", os.path.join(path, filename)

    except ValueError as ve:
        print(ve)
    except IOError as ioe:
        print(f"An error occurred while saving the file: {ioe}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


###############################################################################################################################################################################
def get_label_encoding_dataframe(dataframe, column_name, mapping_dict):
    """
    Apply label encoding to a specified column in a DataFrame using a provided mapping dictionary.

    This function maps the values in the specified column of the DataFrame to new values 
    based on the provided mapping dictionary. It effectively encodes categorical values into 
    numerical values.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column to be encoded.

    column_name : str
        The name of the column in the DataFrame to which label encoding will be applied.

    mapping_dict : dict
        A dictionary mapping original values to their encoded values.

    Returns:
    -------
    pd.Series
        A Series containing the encoded values for the specified column.

    Raises:
    ------
    KeyError: If any value in the column is not found in the mapping dictionary.
    ValueError: If the specified column does not exist in the DataFrame.

    """
    try:
        # Check if the specified column exists in the DataFrame
        if column_name not in dataframe.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

        # Apply label encoding using the mapping dictionary
        encoded_series = dataframe[column_name].map(mapping_dict)

        # Check for missing values in the encoded Series
        if encoded_series.isnull().any():
            raise KeyError("Some values in the column were not found in the mapping dictionary.")

        return encoded_series

    except ValueError as ve:
        print(ve)
    except KeyError as ke:
        print(ke)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



###############################################################################################################################################################################


def get_apply_condition_on_column(dataframe, column_name, condition):
    """
    Apply a condition to a specified column in a DataFrame using a lambda function.

    This function evaluates a specified condition for each element in the given column
    of the DataFrame. The condition should be a valid Python expression that can be evaluated
    for the values in the column.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column on which the condition will be applied.

    column_name : str
        The name of the column in the DataFrame to which the condition will be applied.

    condition : str
        A string representing the condition to evaluate. This should be a valid Python expression
        that uses the variable `x` to refer to the elements of the specified column.

    Returns:
    -------
    pd.Series
        A Series containing the results of applying the condition to the specified column.

    Raises:
    ------
    ValueError: If the specified column does not exist in the DataFrame.
    SyntaxError: If the provided condition is not a valid Python expression.
    TypeError: If the condition cannot be applied to the column values.

    """
    try:
        # Check if the specified column exists in the DataFrame
        if column_name not in dataframe.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        # Apply the condition using a lambda function
        return dataframe[column_name].apply(lambda x: eval(condition))

    except ValueError as ve:
        print(ve)
    except SyntaxError:
        print("The provided condition is not a valid Python expression.")
    except TypeError:
        print("There was an error applying the condition to the column values.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


###############################################################################################################################################################################
def get_two_column_operations(dataframe, columns_1, columns_2, operator):
    """
    Perform a specified arithmetic operation between two columns in a DataFrame.

    This function allows you to perform basic arithmetic operations (addition, subtraction,
    multiplication, and division) on two specified columns of a DataFrame.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the columns on which the operation will be performed.

    columns_1 : str
        The name of the first column in the DataFrame.

    columns_2 : str
        The name of the second column in the DataFrame.

    operator : str
        The arithmetic operator to apply. Supported operators are:
        - '+' for addition
        - '-' for subtraction
        - '*' for multiplication
        - '/' for division

    Returns:
    -------
    pd.Series
        A Series containing the result of the operation.

    Raises:
    ------
    ValueError: If either of the specified columns does not exist in the DataFrame.
    ZeroDivisionError: If division by zero is attempted.
    TypeError: If the operation is attempted on incompatible data types.

    """
    try:
        # Check if the specified columns exist in the DataFrame
        if columns_1 not in dataframe.columns:
            raise ValueError(f"The column '{columns_1}' does not exist in the DataFrame.")
        if columns_2 not in dataframe.columns:
            raise ValueError(f"The column '{columns_2}' does not exist in the DataFrame.")
        
        # Perform the operation based on the specified operator
        if operator == "+":
            return dataframe[columns_1] + dataframe[columns_2]
        elif operator == "-":
            return dataframe[columns_1] - dataframe[columns_2]
        elif operator == "/":
            return dataframe[columns_1] / dataframe[columns_2]
        elif operator == "*":
            return dataframe[columns_1] * dataframe[columns_2]
        else:
            raise ValueError("Unsupported operator. Use '+', '-', '*', or '/'.")
    
    except ValueError as ve:
        print(ve)
    except ZeroDivisionError:
        print("Division by zero encountered.")
    except TypeError:
        print("Type error encountered while performing the operation.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    
###############################################################################################################################################################################    
def get_timedelta_division(dataframe, column, td_type='D'):
    """
    Divide a specified column of a DataFrame containing timedelta values by a specified time unit.

    This function takes a column from a DataFrame that contains timedelta values and divides
    it by a specified time unit (days, hours, minutes, etc.). The result is a numerical representation
    of the timedelta in the specified unit.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column to be divided.

    column : str
        The name of the column in the DataFrame that contains timedelta values.

    td_type : str, optional
        The time unit to divide by. Supported units include:
        - 'D' for days
        - 'h' for hours
        - 'm' for minutes
        - 's' for seconds
        The default is 'D' (days).

    Returns:
    -------
    pd.Series
        A Series containing the result of the division, representing the timedelta in the specified unit.

    Raises:
    ------
    ValueError: If the specified column does not exist in the DataFrame or is not of timedelta type.
    TypeError: If the operation is attempted on incompatible data types.

    """
    try:
        # Check if the specified column exists in the DataFrame
        if column not in dataframe.columns:
            raise ValueError(f"The column '{column}' does not exist in the DataFrame.")
        
        # Check if the column is of timedelta type
        if not pd.api.types.is_timedelta64_dtype(dataframe[column]):
            raise ValueError(f"The column '{column}' is not of timedelta type.")
        
        # Perform the division by the specified time unit
        return dataframe[column] / np.timedelta64(1, td_type)
    
    except ValueError as ve:
        print(ve)
    except TypeError:
        print("Type error encountered while performing the division.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


###############################################################################################################################################################################
def get_replace_value_in_df(dataframe, column, value, replace_with):
    """
    Replace specified values in a given column of a DataFrame with a new value.

    This function searches for all occurrences of a specified value in the specified column
    of a DataFrame and replaces them with a new value.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame in which the values will be replaced.

    column : str
        The name of the column in the DataFrame where the replacement will occur.

    value : scalar
        The value to be replaced in the specified column.

    replace_with : scalar
        The value to replace the specified value with.

    Returns:
    -------
    pd.Series
        A Series containing the updated column after the replacement.

    Raises:
    ------
    ValueError: If the specified column does not exist in the DataFrame.

    """
    try:
        # Check if the specified column exists in the DataFrame
        if column not in dataframe.columns:
            raise ValueError(f"The column '{column}' does not exist in the DataFrame.")
        
        # Replace the specified value with the new value in the specified column
        return dataframe[column].replace(value, replace_with)

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


###############################################################################################################################################################################
def get_validation_unseen_set(dataframe, validation_frac=0.05, sample=False, sample_frac=0.1):
    """
    Split the DataFrame into a training set and an unseen validation set.

    This function takes a DataFrame and splits it into two parts:
    - A training set that includes a specified fraction of the original data,
    - An unseen validation set containing the remaining data that is not included
      in the training set.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be split into training and validation sets.

    validation_frac : float, optional
        The fraction of the DataFrame to be used as the unseen validation set.
        Default is 0.05 (5% of the data).

    sample : bool, optional
        If True, a random sample of the DataFrame is used for the split.
        Default is False, meaning the entire DataFrame is used.

    sample_frac : float, optional
        The fraction of the DataFrame to sample if `sample` is set to True.
        Default is 0.1 (10% of the DataFrame).

    Returns:
    -------
    tuple
        A tuple containing two DataFrames:
        - The training set (DataFrame).
        - The unseen validation set (DataFrame).

    Raises:
    ------
    ValueError: If the validation fraction is not between 0 and 1,
                 or if the sample fraction is not between 0 and 1.

    """
    try:
        # Validate the validation fraction
        if not (0 <= validation_frac < 1):
            raise ValueError("The validation fraction must be between 0 and 1.")

        # Validate the sample fraction if sampling is requested
        if sample and not (0 < sample_frac <= 1):
            raise ValueError("The sample fraction must be between 0 and 1 if sampling is enabled.")

        # Sample the DataFrame if requested
        dataset = dataframe.sample(frac=sample_frac) if sample else dataframe.copy()
        
        # Split into training and unseen validation sets
        data = dataset.sample(frac=(1 - validation_frac), random_state=786)
        data_unseen = dataset.drop(data.index)

        # Reset the index for both sets
        data.reset_index(inplace=True, drop=True)
        data_unseen.reset_index(inplace=True, drop=True)

        return data, data_unseen

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


###############################################################################################################################################################################
def create_sqlit_connection(db_path, db_file):
    """
    Create a database connection to a SQLite database.

    This function attempts to establish a connection to a SQLite database
    located at the specified path and file name. If the connection is
    successful, it prints the SQLite version. The connection is closed
    after the operation.

    Parameters:
    ----------
    db_path : str
        The directory path where the SQLite database file is located.

    db_file : str
        The name of the SQLite database file (including the .db extension).

    Returns:
    -------
    None

    Raises:
    ------
    Error: If there is an issue establishing a connection to the database.

    """
    conn = None
    try:
        # Attempt to create a connection to the SQLite database
        conn = sqlite3.connect(db_path + db_file)
        print(f"Connected to SQLite version: {sqlite3.version}")

    except Error as e:
        print(f"Error connecting to database: {e}")

    finally:
        # Close the connection if it was established
        if conn:
            conn.close()
            print("Database connection closed.")

            
###############################################################################################################################################################################
def get_train_test_set_from_setup():
    """
    Retrieve the training and testing datasets from the configuration setup.

    This function accesses the configuration to retrieve the training
    and testing datasets, including features and target variables.
    It calls the `get_config` function for each dataset variable.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - X_train: Training feature set.
        - y_train: Training target variable.
        - X_test: Testing feature set.
        - y_test: Testing target variable.

    Raises:
    ------
    Exception: If there is an error retrieving any of the datasets.

    """

    try:
        X_train = get_config(variable="X_train")
        y_train = get_config(variable="y_train")
        X_test = get_config(variable="X_test")
        y_test = get_config(variable="y_test")
        
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"Error retrieving datasets: {e}")
        return None, None, None, None  # Return None values if there's an error

###############################################################################################################################################################################
def get_x_y_from_setup():
    """
    Retrieve the feature set (X) and target variable (y) from the configuration setup.

    This function accesses the configuration to retrieve the feature set
    and the target variable for a machine learning model. It calls the 
    `get_config` function for each variable.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - X: Feature set.
        - y: Target variable.

    Raises:
    ------
    Exception: If there is an error retrieving either the feature set or the target variable.

    """

    try:
        X = get_config(variable="X")
        y = get_config(variable="y")
        
        return X, y

    except Exception as e:
        print(f"Error retrieving feature set and target variable: {e}")
        return None, None  # Return None values if there's an error


###############################################################################################################################################################################
def get_transformation_pipeline_from_setup():
    """
    Retrieve the transformation pipeline from the configuration setup.

    This function accesses the configuration to obtain the data transformation 
    pipeline that is used for preprocessing the features before training a model.

    Returns:
    -------
    object
        The transformation pipeline object.

    Raises:
    ------
    Exception: If there is an error retrieving the transformation pipeline.

    """

    try:
        pipeline = get_config(variable="pipeline")
        return pipeline

    except Exception as e:
        print(f"Error retrieving transformation pipeline: {e}")
        return None  # Return None if there's an error


###############################################################################################################################################################################
def check_if_table_has_value(cnx, table_name):
    """
    Check if a specified table exists in the SQLite database.

    This function queries the SQLite database to determine whether a table 
    with the given name exists.

    Parameters:
    ----------
    cnx : sqlite3.Connection
        The SQLite database connection object.

    table_name : str
        The name of the table to check.

    Returns:
    -------
    bool
        True if the table exists, False otherwise.

    Raises:
    ------
    Exception
        If an error occurs while querying the database.

    """
    try:
        # Query to check for the table's existence
        check_table = pd.read_sql(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", cnx
        ).shape[0]
        
        return check_table == 1  # Return True if the table exists, otherwise False

    except Exception as e:
        print(f"Error checking if table '{table_name}' exists: {e}")
        return False  # Return False if an error occurs

    
###############################################################################################################################################################################    
def build_dbs(db_path, db_file_name):
    """
    Create a SQLite database if it does not already exist.

    This function checks if a database file exists at the specified path.
    If it does not exist, it creates a new SQLite database file.

    Parameters:
    ----------
    db_path : str
        The directory path where the database file will be created or checked.

    db_file_name : str
        The name of the SQLite database file.

    Returns:
    -------
    str
        A message indicating whether the database already exists, was created successfully,
        or if an error occurred.

    """
    if os.path.isfile(db_path + db_file_name):
        print("DB Already Exists")
        return "DB Exists"
    else:
        print("Creating Database")
        # Create a database connection to a SQLite database
        conn = None
        try:
            conn = sqlite3.connect(db_path + db_file_name)
            print("New DB Created")
            return "DB Created"
        except Exception as e:
            print(f"Error occurred while creating database: {e}")
            return "Error"
        finally:
            if conn:
                conn.close()

            
###############################################################################################################################################################################
def get_new_data_appended(old_data_directory, new_data_directory, start_data, end_date, append=False):
    """
    Load new data files, filter data based on a date range, and optionally append to existing data.

    This function loads user logs and transaction data from specified directories, filters them based on a given date range and membership criteria, and returns the filtered data. If the `append` parameter is set to `True`, the function appends the filtered data to existing data files.

    Parameters:
    ----------
    old_data_directory : str
        Directory path containing existing data files, including 'members_profile.csv', 'userlogs.csv',
        'transactions_logs.csv', and 'churn_logs.csv'.

    new_data_directory : str
        Directory path containing new data files, specifically 'user_logs_new.csv' and 'transactions_logs_new.csv'.

    start_data : str
        The start date for filtering data in the format 'YYYY-MM-DD'.

    end_date : str
        The end date for filtering data in the format 'YYYY-MM-DD'.

    append : bool, default False
        If True, appends the filtered new data to the existing data and returns the combined result.

    Returns:
    -------
    tuple of pd.DataFrame
        If `append` is False, returns two DataFrames: `march_user_logs` and `march_transactions`,
        containing filtered user logs and transactions within the specified date range.
        
        If `append` is True, returns two DataFrames: `user_logs_combined` and `transactions_combined`,
        containing the original data combined with the filtered new data.

    Raises:
    ------
    FileNotFoundError:
        If any required data file is not found in the specified directory.
    ValueError:
        If data filtering conditions fail or result in empty DataFrames.
    Exception:
        For any other errors encountered during data processing.
    """
    try:
        # Load new data
        user_logs_n, transactions_n = load_data([f"{new_data_directory}user_logs_new.csv",
                                                 f"{new_data_directory}transactions_logs_new.csv"])

        # Load existing data
        members, user_logs, transactions, train = load_data([
            f"{old_data_directory}members_profile.csv",
            f"{old_data_directory}userlogs.csv",
            f"{old_data_directory}transactions_logs.csv",
            f"{old_data_directory}churn_logs.csv"
        ])

        # Get member lists
        members_list = np.unique(list(members['msno']))
        train_members_list = np.unique(list(train['msno']))

        # Filter user logs within the date range and member lists
        user_logs_n['date'] = fix_time_in_df(user_logs_n, 'date', expand=False)
        march_user_logs = user_logs_n[(user_logs_n['date'] > start_data) &
                                      (user_logs_n['date'] < end_date) &
                                      (user_logs_n['msno'].isin(members_list)) &
                                      (user_logs_n['msno'].isin(train_members_list))]

        # Filter transactions within the date range, membership expiration, and member lists
        transactions_n['transaction_date'] = fix_time_in_df(transactions_n, 'transaction_date', expand=False)
        transactions_n['membership_expire_date'] = fix_time_in_df(transactions_n, 'membership_expire_date', expand=False)
        march_transactions = transactions_n[(transactions_n['transaction_date'] > start_data) &
                                            (transactions_n['transaction_date'] < end_date) &
                                            (transactions_n['membership_expire_date'] < '2017-12-31') &
                                            (transactions_n['msno'].isin(members_list)) &
                                            (transactions_n['msno'].isin(train_members_list))]

        # Append data if specified
        if not append:
            return march_user_logs, march_transactions
        else:
            user_logs_combined = user_logs.append(march_user_logs, ignore_index=True)
            transactions_combined = transactions.append(march_transactions, ignore_index=True)
            return user_logs_combined, transactions_combined

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except ValueError as ve:
        print(f"Data processing error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
   
###############################################################################################################################################################################
def load_data_from_source(db_path, db_file_name, drfit_db_name, 
                          old_data_directory, new_data_directory,
                          run_on='old', start_data='2017-03-01', end_date='2017-03-31',
                          append=True):
    """
    Load data from specified source directories into a SQLite database.

    This function checks if the specified process flag indicates that 
    loading data is necessary. Depending on the `run_on` parameter, 
    it loads either old or new data into the database, appending data 
    if specified.

    Parameters:
    ----------
    db_path : str
        The directory path where the SQLite database file is located.

    db_file_name : str
        The name of the SQLite database file.

    drift_db_name : str
        The name of the drift database used to check process flags.

    old_data_directory : str
        The directory path for old data source files.

    new_data_directory : str
        The directory path for new data source files.

    run_on : str, optional
        Specifies whether to run on 'old' or 'new' data. Default is 'old'.

    start_data : str, optional
        Start date for filtering new data. Default is '2017-03-01'.

    end_date : str, optional
        End date for filtering new data. Default is '2017-03-31'.

    append : bool, optional
        If True, appends new data to existing data. Default is True.

    Returns:
    -------
    str
        Message indicating the status of the data loading operation.

    """
    
    # Get process flag dataframe
    cnx_drift = sqlite3.connect(db_path + drfit_db_name)
    process_flags = pd.read_sql('SELECT * FROM process_flags', cnx_drift)
    
    if process_flags['load_data'][0] == 1:
        cnx = sqlite3.connect(db_path + db_file_name)

        if run_on == 'old':
            print("Running on OLD Data") 
            # Load old data into the database
            if not check_if_table_has_value(cnx, 'train'):
                print("Table Doesn't Exist - train, Building")
                train = load_data([f"{old_data_directory}churn_logs.csv"])[0]
                train.to_sql(name='train', con=cnx, if_exists='replace', index=False)

            if not check_if_table_has_value(cnx, 'user_logs'):
                print("Table Doesn't Exist - user_logs, Building")
                user_logs = load_data([f"{old_data_directory}userlogs.csv"])[0]
                user_logs, pre_size, post_size = compress_dataframes([user_logs])[0]
                print(f"user_logs DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                user_logs.to_sql(name='user_logs', con=cnx, if_exists='replace', index=False)

            if not check_if_table_has_value(cnx, 'transactions'):
                print("Table Doesn't Exist - transactions, Building")
                transactions = load_data([f"{old_data_directory}transactions_logs.csv"])[0]
                transactions, pre_size, post_size = compress_dataframes([transactions])[0]
                print(f"transactions DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                transactions.to_sql(name='transactions', con=cnx, if_exists='replace', index=False)

            if not check_if_table_has_value(cnx, 'members'):
                print("Table Doesn't Exist - members, Building")
                members = load_data([f"{old_data_directory}members_profile.csv"])[0]
                members, pre_size, post_size = compress_dataframes([members])[0]
                print(f"members DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                members.to_sql(name='members', con=cnx, if_exists='replace', index=False)

        elif run_on == 'new':
            if append:
                print("Running on New Data") 
                # Append new data to existing data
                march_user_logs, march_transactions = get_new_data_appended(old_data_directory, new_data_directory, start_data, end_date)

                if not check_if_table_has_value(cnx, 'train'):
                    print("Table Doesn't Exist - train, Building")
                    train = load_data([f"{old_data_directory}churn_logs.csv"])[0]
                    train.to_sql(name='train', con=cnx, if_exists='replace', index=False)

                if not check_if_table_has_value(cnx, 'user_logs'):
                    print("Table Doesn't Exist - user_logs, Building")
                    user_logs = load_data([f"{old_data_directory}userlogs.csv"])[0]
                    user_logs['date'] = fix_time_in_df(user_logs, 'date', expand=False)
                    user_logs_appended = user_logs.append(march_user_logs)
                    user_logs, pre_size, post_size = compress_dataframes([user_logs_appended])[0]
                    print(user_logs.head())
                    print(f"user_logs DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                    user_logs.to_sql(name='user_logs', con=cnx, if_exists='replace', index=False)

                if not check_if_table_has_value(cnx, 'transactions'):
                    print("Table Doesn't Exist - transactions, Building")
                    transactions = load_data([f"{old_data_directory}transactions_logs.csv"])[0]
                    transactions['transaction_date'] = fix_time_in_df(transactions, 'transaction_date', expand=False)
                    transactions['membership_expire_date'] = fix_time_in_df(transactions, 'membership_expire_date', expand=False)
                    transactions_appended = transactions.append(march_transactions)
                    transactions, pre_size, post_size = compress_dataframes([transactions_appended])[0]
                    print(f"transactions DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                    transactions.to_sql(name='transactions', con=cnx, if_exists='replace', index=False)

                if not check_if_table_has_value(cnx, 'members'):
                    print("Table Doesn't Exist - members, Building")
                    members = load_data([f"{old_data_directory}members_profile.csv"])[0]
                    members, pre_size, post_size = compress_dataframes([members])[0]
                    print(f"members DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                    members.to_sql(name='members', con=cnx, if_exists='replace', index=False)

            else:
                print("Running on New Data without Append.") 
                # Load new data without appending
                if not check_if_table_has_value(cnx, 'train'):
                    print("Table Doesn't Exist - train, Building")
                    train = load_data([f"{new_data_directory}churn_logs_new.csv"])[0]
                    train.to_sql(name='train', con=cnx, if_exists='replace', index=False)

                if not check_if_table_has_value(cnx, 'user_logs'):
                    print("Table Doesn't Exist - user_logs, Building")
                    user_logs = load_data([f"{new_data_directory}user_logs_new.csv"])[0]
                    user_logs, pre_size, post_size = compress_dataframes([user_logs])[0]
                    print(f"user_logs DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                    user_logs.to_sql(name='user_logs', con=cnx, if_exists='replace', index=False)

                if not check_if_table_has_value(cnx, 'transactions'):
                    print("Table Doesn't Exist - transactions, Building")
                    transactions = load_data([f"{new_data_directory}transactions_logs_new.csv"])[0]
                    transactions, pre_size, post_size = compress_dataframes([transactions])[0]
                    print(f"transactions DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                    transactions.to_sql(name='transactions', con=cnx, if_exists='replace', index=False)

                if not check_if_table_has_value(cnx, 'members'):
                    print("Table Doesn't Exist - members, Building")
                    members = load_data([f"{new_data_directory}members_profile_new.csv"])[0]
                    members, pre_size, post_size = compress_dataframes([members])[0]
                    print(f"members DF before compress was in MB: {pre_size}, and after compress: {post_size}")
                    members.to_sql(name='members', con=cnx, if_exists='replace', index=False)

        cnx.close()
        return "Writing to DataBase Done or Data Already was in Table. Check Logs."
                
    else:
        print("Skipping..... Not required")


###############################################################################################################################################################################
def get_membership_data_transform(db_path, db_file_name, drfit_db_name):
    """
    Transforms and saves membership data from the 'members' table into a new table 'members_final'.
    
    This function performs the following operations:
    1. Connects to the specified SQLite databases.
    2. Checks a process flag to determine if membership data transformation is required.
    3. If transformation is required and the 'members_final' table does not exist:
        - Reads data from the 'members' table.
        - Fills missing values in the 'gender' column with "others".
        - Encodes the 'gender' column using label encoding.
        - Converts the 'registered_via' and 'city' columns to string data types.
        - Fixes the 'registration_init_time' column.
        - Applies a conditional transformation to the 'bd' column based on the average age.
        - Saves the transformed data into the 'members_final' table.
    4. Returns a message indicating whether the data was transformed and saved, or if it was already transformed.
    
    Parameters:
        db_path (str): The file path to the database.
        db_file_name (str): The name of the database file.
        drfit_db_name (str): The name of the drift database file.
        
    Returns:
        str: A message indicating the result of the transformation.
    """
    
    try:
        # Establish connections to the databases
        cnx = sqlite3.connect(db_path + db_file_name)
        cnx_drift = sqlite3.connect(db_path + drfit_db_name)

        # Read the process flags
        process_flags = pd.read_sql('select * from process_flags', cnx_drift)

        if process_flags['process_members'][0] == 1:
            if not check_if_table_has_value(cnx, 'members_final'):
                # Load the members data
                members = pd.read_sql('select * from members', cnx)

                # Data transformations
                members['gender'] = get_fill_na_dataframe(members, 'gender', value="others")
                gender_mapping = {'male': 0, 'female': 1, 'others': 2}
                members['gender'] = get_label_encoding_dataframe(members, 'gender', gender_mapping)
                members['registered_via'] = get_convert_column_dtype(members, 'registered_via', data_type='str')
                members['city'] = get_convert_column_dtype(members, 'city', data_type='str')
                members['registration_init_time'] = fix_time_in_df(members, 'registration_init_time', expand=False)
                
                average_age = round(members['bd'].mean(), 2)
                condition = f"{average_age} if (x <= 0 or x > 100) else x"
                members['bd'] = get_apply_condiiton_on_column(members, 'bd', condition)

                # Save the transformed data into the new table
                members.to_sql(name='members_final', con=cnx, if_exists='replace', index=False)

                return "Membership Data is Transformed and Saved into members_final"
            return "Membership Data is already Transformed and Saved into members_final"
        else:
            print("Not Required......Skipping")
    except Exception as e:
        return f"An error occurred during membership data transformation: {str(e)}"
    finally:
        # Ensure that database connections are closed
        if 'cnx' in locals():
            cnx.close()
        if 'cnx_drift' in locals():
            cnx_drift.close()


###############################################################################################################################################################################
def get_transaction_data_transform(db_path, db_file_name, drfit_db_name):
    """
    Transforms and saves transaction data from the 'transactions' table into a new table 'transactions_features_final'.
    
    This function performs the following operations:
    1. Connects to the specified SQLite databases.
    2. Checks a process flag to determine if transaction data transformation is required.
    3. If transformation is required and the 'transactions_features_final' table does not exist:
        - Reads data from the 'transactions' table.
        - Fixes date columns for 'transaction_date' and 'membership_expire_date'.
        - Calculates discounts and identifies if a discount was applied.
        - Computes the amount paid per day and replaces infinite values.
        - Calculates membership duration and checks if it's greater than 30 days.
        - Aggregates transaction features for each member.
        - Saves the transformed data into the 'transactions_features_final' table.
    4. Returns a message indicating whether the data was transformed and saved or if it was already transformed.
    
    Parameters:
        db_path (str): The file path to the database.
        db_file_name (str): The name of the database file.
        drfit_db_name (str): The name of the drift database file.
        
    Returns:
        str: A message indicating the result of the transformation.
    """
    
    try:
        # Establish connections to the databases
        cnx = sqlite3.connect(db_path + db_file_name)
        cnx_drift = sqlite3.connect(db_path + drfit_db_name)

        # Read the process flags
        process_flags = pd.read_sql('select * from process_flags', cnx_drift)

        if process_flags['process_transactions'][0] == 1:
            if not check_if_table_has_value(cnx, 'transactions_features_final'):
                # Load the transactions data
                transactions = pd.read_sql('select * from transactions', cnx)

                # Data transformations
                transactions['transaction_date'] = fix_time_in_df(transactions, 'transaction_date', expand=False)
                transactions['membership_expire_date'] = fix_time_in_df(transactions, 'membership_expire_date', expand=False)

                transactions['discount'] = get_two_column_operations(transactions, 'plan_list_price', 'actual_amount_paid', "-")
                condition = "1 if x > 0 else 0"
                transactions['is_discount'] = get_apply_condiiton_on_column(transactions, 'discount', condition)

                transactions['amt_per_day'] = get_two_column_operations(transactions, 'actual_amount_paid', 'payment_plan_days', "/")
                transactions['amt_per_day'] = get_replace_value_in_df(transactions, 'amt_per_day', [np.inf, -np.inf], replace_with=0)

                transactions['membership_duration'] = get_two_column_operations(transactions, 'membership_expire_date', 'transaction_date', "-")
                transactions['membership_duration'] = get_timedelta_division(transactions, "membership_duration", td_type='D')
                transactions['membership_duration'] = get_convert_column_dtype(transactions, 'membership_duration', data_type='int')

                condition = "1 if x > 30 else 0"
                transactions['more_than_30'] = get_apply_condiiton_on_column(transactions, 'membership_duration', condition)

                # Define aggregation dictionary
                agg = {
                    'payment_method_id': ['count', 'nunique'],  # Number of transactions per payment method
                    'payment_plan_days': ['mean', 'nunique'],    # Average plan duration and unique plan changes
                    'plan_list_price': 'mean',                    # Average charged amount
                    'actual_amount_paid': 'mean',                 # Average amount paid
                    'is_auto_renew': ['mean', 'max'],             # Auto-renew status changes
                    'transaction_date': ['min', 'max', 'count'],  # First and last transaction
                    'membership_expire_date': 'max',              # Last membership expiration date
                    'is_cancel': ['mean', 'max'],                 # Cancellation status changes
                    'discount': 'mean',                           # Average discount given
                    'is_discount': ['mean', 'max'],               # Discount flag changes
                    'amt_per_day': 'mean',                        # Average daily amount spent
                    'membership_duration': 'mean',                 # Average membership duration
                    'more_than_30': 'sum'                         # Flags for duration greater than 30
                }

                # Perform aggregation
                transactions_features = get_groupby(transactions, by_column='msno', agg_dict=agg, agg_func='mean', simple_agg_flag=False, reset_index=True)
                transactions_features.columns = transactions_features.columns.get_level_values(0) + '_' + transactions_features.columns.get_level_values(1)
                transactions_features.rename(columns={
                    'msno_': 'msno',
                    'payment_plan_days_nunique': 'change_in_plan',
                    'payment_method_id_count': 'total_payment_channels',
                    'payment_method_id_nunique': 'change_in_payment_methods',
                    'is_cancel_max': 'is_cancel_change_flag',
                    'is_auto_renew_max': 'is_autorenew_change_flag',
                    'transaction_date_count': 'total_transactions'
                }, inplace=True)

                # Save the transformed data into the new table
                transactions_features.to_sql(name='transactions_features_final', con=cnx, if_exists='replace', index=False)

                return "Transactions Data is Transformed and Saved into transactions_features_final"
            return "Transactions Data is already Transformed and Saved into transactions_features_final"
        
        else:
            print("Not Required......Skipping")
    
    except Exception as e:
        return f"An error occurred during transaction data transformation: {str(e)}"
    
    finally:
        # Ensure that database connections are closed
        if 'cnx' in locals():
            cnx.close()
        if 'cnx_drift' in locals():
            cnx_drift.close()


###############################################################################################################################################################################
def get_user_data_transform(db_path, db_file_name, drfit_db_name):
    """
    Transforms and saves user log data from the 'user_logs' table into a new table 'user_logs_features_final'.
    
    This function performs the following operations:
    1. Connects to the specified SQLite databases.
    2. Checks a process flag to determine if user log data transformation is required.
    3. If transformation is required and the 'user_logs_features_final' table does not exist:
        - Reads data from the 'user_logs' table.
        - Fixes date columns for the 'date' field.
        - Applies logarithmic transformation to selected columns to correct skewness.
        - Aggregates user log features including login frequency and last login date.
        - Merges the aggregated results into a final dataframe.
        - Saves the transformed data into the 'user_logs_features_final' table.
    4. Returns a message indicating whether the data was transformed and saved or if it was already transformed.
    
    Parameters:
        db_path (str): The file path to the database.
        db_file_name (str): The name of the database file.
        drfit_db_name (str): The name of the drift database file.
        
    Returns:
        str: A message indicating the result of the transformation.
    """
    
    try:
        # Establish connections to the databases
        cnx = sqlite3.connect(db_path + db_file_name)
        cnx_drift = sqlite3.connect(db_path + drfit_db_name)

        # Read the process flags
        process_flags = pd.read_sql('select * from process_flags', cnx_drift)

        if process_flags['process_userlogs'][0] == 1:
            if not check_if_table_has_value(cnx, 'user_logs_features_final'):
                # Load the user logs data
                user_logs = pd.read_sql('select * from user_logs', cnx)

                # Data transformations
                user_logs['date'] = fix_time_in_df(user_logs, column_name='date', expand=False)
                user_logs_transformed = get_fix_skew_with_log(user_logs, 
                    ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'], 
                    replace_inf=True, replace_inf_with=0)

                # Group by 'msno' and compute mean for transformed data
                user_logs_transformed_base = get_groupby(user_logs_transformed, 'msno', 
                    agg_dict=None, agg_func='mean', simple_agg_flag=True, reset_index=True)

                # Aggregating login frequency and last login date
                agg_dict = {'date': ['count', 'max']}
                user_logs_transformed_dates = get_groupby(user_logs_transformed, 'msno', 
                    agg_dict=agg_dict, agg_func='mean', simple_agg_flag=False, reset_index=True)
                user_logs_transformed_dates.columns = user_logs_transformed_dates.columns.droplevel()
                user_logs_transformed_dates.rename(columns={'count': 'login_freq', 'max': 'last_login'}, inplace=True)
                user_logs_transformed_dates.reset_index(drop=True, inplace=True)

                # Merging base and date features
                user_logs_final = get_merge(user_logs_transformed_base, user_logs_transformed_dates, on='msno')

                # Save the transformed data into the new table
                user_logs_final.to_sql(name='user_logs_features_final', con=cnx, if_exists='replace', index=False)

                return "User logs Data is Transformed and Saved into user_logs_features_final"
            return "User logs Data is already Transformed and Saved into user_logs_features_final"
        
        else:
            print("Not Required......Skipping")
    
    except Exception as e:
        return f"An error occurred during user logs data transformation: {str(e)}"
    
    finally:
        # Ensure that database connections are closed
        if 'cnx' in locals():
            cnx.close()
        if 'cnx_drift' in locals():
            cnx_drift.close()


###############################################################################################################################################################################
def get_final_data_merge(db_path, db_file_name, drfit_db_name):
    """
    Merges data from multiple tables into a final features table and saves it to the database.

    This function performs the following operations:
    1. Connects to the specified SQLite databases.
    2. Checks a process flag to determine if merging of user logs data is required.
    3. If required and the 'final_features_v01' table does not exist:
        - Reads data from 'members_final', 'transactions_features_final', 'user_logs_features_final', and 'train' tables.
        - Merges the dataframes on the 'msno' column.
        - Saves the merged dataframe into the 'final_features_v01' table.
    4. Returns a message indicating whether the merge was performed or if the table already exists.

    Parameters:
        db_path (str): The file path to the database.
        db_file_name (str): The name of the database file.
        drfit_db_name (str): The name of the drift database file.

    Returns:
        str: A message indicating the result of the merge operation.
    """
    
    try:
        # Establish connections to the databases
        cnx = sqlite3.connect(db_path + db_file_name)
        cnx_drift = sqlite3.connect(db_path + drfit_db_name)

        # Read the process flags
        process_flags = pd.read_sql('select * from process_flags', cnx_drift)

        if process_flags['process_userlogs'][0] == 1:
            if not check_if_table_has_value(cnx, 'final_features_v01'):
                print("Final Merge Doesn't Exist in DB") 
                
                # Load data from various tables
                members_final = pd.read_sql('select * from members_final', cnx)
                transactions_final = pd.read_sql('select * from transactions_features_final', cnx)
                user_logs_final = pd.read_sql('select * from user_logs_features_final', cnx)
                train = pd.read_sql('select * from train', cnx)

                # Merge dataframes on 'msno'
                train_df_v01 = get_merge(members_final, train, on='msno', axis=1, how='inner')
                train_df_v02 = get_merge(train_df_v01, transactions_final, on='msno', axis=1, how='inner')
                train_df_final = get_merge(train_df_v02, user_logs_final, on='msno', axis=1, how='inner')

                # Save the merged dataframe to the final features table
                train_df_final.to_sql(name='final_features_v01', con=cnx, if_exists='replace', index=False)

                return "All Data is Merged and Saved into final_features_v01"
            else:
                return "Final Merged Already Performed and Available in DB" 
        
        else:
            print("Not Required......Skipping")
    
    except Exception as e:
        return f"An error occurred during the final data merge: {str(e)}"
    
    finally:
        # Ensure that database connections are closed
        if 'cnx' in locals():
            cnx.close()
        if 'cnx_drift' in locals():
            cnx_drift.close()

###############################################################################################################################################################################
def get_data_prepared_for_modeling(db_path, db_file_name, drfit_db_name, scale_method='standard', date_columns=None, 
                                    corr_threshold=0.90, drop_corr=False, date_transformation=True):
    """
    Prepares data for modeling by scaling features, handling correlations, and transforming date columns.

    This function performs the following operations:
    1. Connects to the specified SQLite databases.
    2. Checks a process flag to determine if data preparation is required.
    3. If required and the 'X' and 'y' tables do not exist:
        - Reads data from the 'final_features_v01' table.
        - Calculates and drops features with high correlation.
        - Transforms specified date columns into separate features.
        - Scales numeric features.
        - Saves the prepared features (X) and target variable (y) into the database.
    4. Returns a message indicating the result of the operation.

    Parameters:
        db_path (str): The file path to the database.
        db_file_name (str): The name of the database file.
        drfit_db_name (str): The name of the drift database file.
        scale_method (str): The scaling method to use ('standard' or others).
        date_columns (list): List of date column names to transform.
        corr_threshold (float): The correlation threshold for dropping features.
        drop_corr (bool): Whether to drop features with high correlation.
        date_transformation (bool): Whether to transform date columns.

    Returns:
        str: A message indicating the result of the data preparation operation.
    """
    
    try:
        # Establish connections to the databases
        cnx = sqlite3.connect(db_path + db_file_name)
        cnx_drift = sqlite3.connect(db_path + drfit_db_name)

        # Read the process flags
        process_flags = pd.read_sql('select * from process_flags', cnx_drift)

        if process_flags['Data_Preparation'][0] == 1:
            if not check_if_table_has_value(cnx, 'X') and not check_if_table_has_value(cnx, 'y'):
                dataframe = pd.read_sql('select * from final_features_v01', cnx)

                # Select only numeric columns for correlation calculation
                numeric_dataframe = dataframe.select_dtypes(include=[np.number])

                # Create correlation matrix
                corr_matrix = numeric_dataframe.corr().abs()
                # Select upper triangle of correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                # Find features with correlation greater than the specified threshold
                to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
                print("Dropping columns with high correlation:", to_drop)

                # Drop features if specified
                if drop_corr:
                    dataframe.drop(to_drop, axis=1, inplace=True)

                print("Remaining columns after dropping:", len(dataframe.columns))

                if date_transformation and date_columns is not None:
                    # Date transformation
                    features = ["day", "month", "year", "weekday"]
                    date_data = dataframe[date_columns].copy()

                    for eachcol in date_data:
                        date_data[eachcol] = pd.to_datetime(date_data[eachcol], errors='coerce')  # Convert to datetime
                        # Extract features
                        for eachfeature in features:
                            col_name = f"{eachcol}_{eachfeature}"
                            if eachfeature == 'day':
                                date_data[col_name] = date_data[eachcol].dt.day
                            elif eachfeature == 'month':
                                date_data[col_name] = date_data[eachcol].dt.month
                            elif eachfeature == 'year':
                                date_data[col_name] = date_data[eachcol].dt.year
                            elif eachfeature == 'weekday':
                                date_data[col_name] = date_data[eachcol].dt.weekday
                    date_data.drop(date_columns, axis=1, inplace=True)
                    final_date = pd.get_dummies(date_data, drop_first=True, dtype='int16')

                # Scaling
                column_to_scale = dataframe.select_dtypes(include=['float64', 'int64', 'float32']).columns.drop('is_churn')
                transformer = StandardScaler().fit(dataframe[column_to_scale])
                scaled_data = pd.DataFrame(transformer.transform(dataframe[column_to_scale]), columns=column_to_scale)

                # Combining
                if date_transformation:
                    final_df = pd.concat([scaled_data, final_date, dataframe['is_churn']], axis=1)
                else:
                    print("Doing Feature without Dates")
                    final_df = pd.concat([scaled_data, dataframe['is_churn']], axis=1)

                # Splitting X, y
                X = final_df.drop(['is_churn'], axis=1)
                y = final_df[['is_churn']]
                index_df = dataframe[['msno']].copy()
                index_df['index_for_map'] = index_df.index

                # Writing to SQL
                X.to_sql(name='X', con=cnx, if_exists='replace', index=False)
                y.to_sql(name='y', con=cnx, if_exists='replace', index=False)
                index_df.to_sql(name='index_msno_mapping', con=cnx, if_exists='replace', index=False)
                return "X & Y written in the database"
            else:
                return "X & Y already exist in Data."
        else:
            print("Not Required... Skipping")
    
    except Exception as e:
        return f"An error occurred during data preparation: {str(e)}"
    
    finally:
        # Ensure that database connections are closed
        if 'cnx' in locals():
            cnx.close()
        if 'cnx_drift' in locals():
            cnx_drift.close()


###############################################################################################################################################################################
# def get_data_prepared_for_modeling(db_path, db_file_name, drfit_db_name, scale_method='standard', date_columns=None, corr_threshold=0.90, drop_corr=False, date_transformation=True):
#     """
#     Prepares the data for modeling by performing correlation analysis, date transformations, 
#     and scaling of numeric features, then saves processed data to the database.
#     """
#     cnx = sqlite3.connect(db_path + db_file_name)
#     cnx_drift = sqlite3.connect(db_path + drfit_db_name)
#     process_flags = pd.read_sql('SELECT * FROM process_flags', cnx_drift)

#     if process_flags['Data_Preparation'][0] == 1:
#         if not check_if_table_has_value(cnx, 'X') and not check_if_table_has_value(cnx, 'y'):
#             dataframe = pd.read_sql('SELECT * FROM final_features_v01', cnx)

#             # Select only numeric columns for correlation calculation
#             numeric_dataframe = dataframe.select_dtypes(include=[np.number])
#             corr_matrix = numeric_dataframe.corr().abs()
#             upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#             to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
#             print("Dropping columns with high correlation:", to_drop)

#             # Drop features if specified
#             if drop_corr:
#                 dataframe.drop(to_drop, axis=1, inplace=True)
#             print("Remaining columns after dropping:", len(dataframe.columns))

#             if date_transformation and date_columns is not None:
#                 date_data = dataframe[date_columns].copy()
#                 for eachcol in date_data:
#                     date_data[eachcol] = pd.to_datetime(date_data[eachcol], errors='coerce')
#                     for eachfeature in ["day", "month", "year", "weekday"]:
#                         col_name = f"{eachcol}_{eachfeature}"
#                         date_data[col_name] = getattr(date_data[eachcol].dt, eachfeature)
#                 date_data.drop(date_columns, axis=1, inplace=True)
#                 final_date = pd.get_dummies(date_data, drop_first=True, dtype='int16')

#             # Scaling
#             column_to_scale = dataframe.select_dtypes(include=['float64', 'int64', 'float32']).columns.drop('is_churn')
#             transformer = StandardScaler().fit(dataframe[column_to_scale])
#             scaled_data = pd.DataFrame(transformer.transform(dataframe[column_to_scale]), columns=column_to_scale)

#             final_df = pd.concat([scaled_data, final_date, dataframe['is_churn']], axis=1) if date_transformation else pd.concat([scaled_data, dataframe['is_churn']], axis=1)

#             X = final_df.drop(['is_churn'], axis=1)
#             y = final_df[['is_churn']]
#             index_df = dataframe[['msno']].assign(index_for_map=dataframe.index)

#             # Writing to SQL
#             X.to_sql(name='X', con=cnx, if_exists='replace', index=False)
#             y.to_sql(name='y', con=cnx, if_exists='replace', index=False)
#             index_df.to_sql(name='index_msno_mapping', con=cnx, if_exists='replace', index=False)
#             return "X & Y written in the database"
#         else:
#             return "X & Y already exist in Data."
#     else:
#         print("Not Required... Skipping")
###############################################################################################################################################################################
def get_train_model(db_path, db_file_name, drfit_db_name):
    """
    Trains a LightGBM model using data from the database and logs the results to MLflow.
    """
    mlflow.set_tracking_uri("http://mlflow:6006")
    cnx_drift = sqlite3.connect(db_path + drfit_db_name)
    process_flags = pd.read_sql('SELECT * FROM process_flags', cnx_drift)

    if process_flags['Data_Preparation'][0] == 1:
        cnx = sqlite3.connect(db_path + db_file_name)
        X = pd.read_sql('SELECT * FROM X', cnx)
        y = pd.read_sql('SELECT * FROM y', cnx).values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        model_config = {
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
        }

        with mlflow.start_run(run_name='run_LightGB_withoutHPTune') as run:
            clf = lgb.LGBMClassifier(**model_config)
            clf.fit(X_train, y_train)
            mlflow.sklearn.log_model(clf, artifact_path="models", registered_model_name='LightGBM')
            mlflow.log_params(model_config)

            # Predict the results on the test dataset
            y_pred = clf.predict(X_test)

            # Log metrics
            acc = accuracy_score(y_test, y_pred)
            conf_mat = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)

            tn, fp, fn, tp = conf_mat.ravel()
            mlflow.log_metric("True Negative", tn)
            mlflow.log_metric("False Positive", fp)
            mlflow.log_metric("False Negative", fn)
            mlflow.log_metric("True Positive", tp)

            print("Inside MLflow Run with id {}".format(run.info.run_uuid))
    else:
        print("Not Required... Skipping")

###############################################################################################################################################################################

def setup_mlflow_experiment(mlflow_tracking_uri, experiment_name):
    """
    Set up an MLflow experiment.

    Parameters:
    - mlflow_tracking_uri (str): The tracking URI for MLflow.
    - experiment_name (str): The name of the experiment to create or set.

    Returns:
    - bool: True if the experiment was created, False if it already existed.
    """
    if not isinstance(mlflow_tracking_uri, str) or not mlflow_tracking_uri:
        raise ValueError("mlflow_tracking_uri must be a non-empty string.")
    
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("experiment_name must be a non-empty string.")

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    logging.info("Creating MLflow experiment")
    try:
        mlflow.create_experiment(experiment_name)
        logging.info(f"Experiment '{experiment_name}' created successfully.")
        return True
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            logging.warning(f"Experiment '{experiment_name}' already exists. Using the existing experiment.")
            return False
        else:
            logging.error(f"Failed to create experiment '{experiment_name}': {e}")
            raise  # Re-raise the exception for further handling

    mlflow.set_experiment(experiment_name)

###############################################################################################################################################################################
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_mlflow_experiment(mlflow_tracking_uri, experiment_name):
#     """Sets up the MLflow experiment."""
#     # (Function implementation here as before)

def load_data_X_y(db_path, db_file_name):
    """Loads data from the SQLite database."""
    with sqlite3.connect(db_path + db_file_name) as cnx:
        X = pd.read_sql('SELECT * FROM X', cnx)
        y = pd.read_sql('SELECT * FROM y', cnx).values.ravel()
    return X, y

def train_model(X, y):
    """Trains the LightGBM model with hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Define your categorical features if there are any
    categoricals = []
    indexes_of_categories = [X_train.columns.get_loc(col) for col in categoricals]

    gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y)

    gridParams = {
        'learning_rate': [0.005, 0.01, 0.1],
        'n_estimators': [8, 16, 24, 50],
        'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt', 'dart'],
        'objective': ['binary'],
        'max_bin': [255, 510],
        'random_state': [500],
        'colsample_bytree': [0.64, 0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4],
        'max_depth': [1, 3, 5]
    }

    model_params = {
        'objective': 'binary',
        'num_boost_round': 200,
        'metric': 'f1',
        'categorical_feature': indexes_of_categories,
        'verbose': -1,
        'force_row_wise': True
    }

    lgb_estimator = lgb.LGBMClassifier()
    lgb_estimator.set_params(**model_params)

    gsearch = BayesSearchCV(estimator=lgb_estimator, search_spaces=gridParams, cv=gkf, 
                            n_iter=32, random_state=0, n_jobs=-1, verbose=1, scoring='f1')
    return gsearch.fit(X, y)

def log_model_to_mlflow(run_name, model, X_test, y_test):
    """Logs the model and its metrics to MLflow."""
    with mlflow.start_run(run_name=run_name) as run:
        y_pred = model.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(model, registered_model_name='LightGBM', artifact_path='models')

        # Log params
        model_params = model.best_estimator_.get_params()
        for param, value in model_params.items():
            mlflow.log_param(param, value)

        # Log metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('false_negatives', cm[1][0])
        mlflow.log_metric('true_negatives', cm[0][0])

        logging.info(f"MLflow Run ID: {run.info.run_uuid}")

def get_train_model_hptune(db_path, db_file_name, drfit_db_name):
    """Main function to get and train the model with hyperparameter tuning."""
    mlflow_tracking_uri = "http://mlflow:6006"
    mlflow_experiment_name = "Model_Building_Pipeline"
    experiment_name = f"{mlflow_experiment_name}_{date.today().strftime('%d_%m_%Y_%H_%M_%S')}"
    
    setup_mlflow_experiment(mlflow_tracking_uri, experiment_name)
    
    try:
        # Load drift flags
        with sqlite3.connect(db_path + drfit_db_name) as cnx_drift:
            process_flags = pd.read_sql('SELECT * FROM process_flags', cnx_drift)

        if process_flags['Model_Training_hpTunning'][0] == 1:
            X, y = load_data_X_y(db_path, db_file_name)
            logging.info(f"Loaded data with shapes: X={X.shape}, y={y.shape}")

            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            model = train_model(X_train, y_train)
            logging.info("Model training complete.")

            # Log the best model and metrics
            log_model_to_mlflow(f"LGBM_Bayes_Search_{int(time.time())}", model, X_test, y_test)

        else:
            logging.info("Model training is not required. Skipping.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        mlflow.end_run(status='FAILED')


###############################################################################################################################################################################
# def get_predict(db_path,db_file_name,ml_flow_path,drift_db_name):
#     cnx_drift = sqlite3.connect(db_path+drift_db_name)
#     process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
#     if process_flags['Prediction'][0] == 1:
#         mlflow_tracking_uri = "http://mlflow:6006"
#         cnx = sqlite3.connect(db_path+db_file_name)
#         logged_model = ml_flow_path
#         # Load model as a PyFuncModel.
#         loaded_model = mlflow.sklearn.load_model(logged_model)
#         # Predict on a Pandas DataFrame.
#         X = pd.read_sql('select * from X', cnx)
#         predictions_proba = loaded_model.predict_proba(pd.DataFrame(X))
#         predictions = loaded_model.predict(pd.DataFrame(X))
#         pred_df = X.copy()
         
        
#         pred_df['churn'] = predictions
#         pred_df[["Prob of Not Churn","Prob of Churn"]] = predictions_proba
        

#         index_msno_mapping = pd.read_sql('select * from index_msno_mapping', cnx)
#         pred_df['index_for_map'] = pred_df.index
#         final_pred_df = pred_df.merge(index_msno_mapping, on='index_for_map') 
#         final_pred_df.to_sql(name='predictions', con=cnx,if_exists='replace',index=False)
#         print (pd.DataFrame(predictions_proba,columns=["Prob of Not Churn","Prob of Churn"]).head()) 
#         pd.DataFrame(final_pred_df,columns=["Prob of Not Churn","Prob of Churn"]).to_sql(name='Final_Predictions', con=cnx,if_exists='replace',index=False)
#         return final_pred_df

#     else:
#         print("Not Required......Skipping")

##########################################################################################################################################################################################
# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_predict_mlflow_server(db_path: str, db_file_name: str, drift_db_name: str):
    try:
        with sqlite3.connect(db_path + drift_db_name) as cnx_drift:
            process_flags = pd.read_sql('SELECT * FROM process_flags', cnx_drift)

        if process_flags['Prediction'].iloc[0] == 1:
            mlflow_serving_uri = "http://mlflow_serve:5000/invocations"

            with sqlite3.connect(db_path + db_file_name) as cnx:
                X = pd.read_sql('SELECT * FROM X', cnx)
                logger.info(f"Loaded data for prediction: {X.head()}")

                response = requests.post(mlflow_serving_uri, json={"dataframe_records": X.to_dict(orient="records")})
                
                if response.status_code == 200:
                    predictions = response.json()
                    logger.info(f"Received predictions: {predictions}")
                    
                    pred_df = X.copy()
                    pred_df['churn'] = predictions['predictions']  # Adjust based on the response structure
                    
                    pred_df.to_sql(name='predictions', con=cnx, if_exists='replace', index=False)
                    logger.info(f"Predictions saved: {pred_df[['churn']].head()}")

                    return pred_df
                else:
                    logger.error(f"Error in prediction request: {response.status_code} - {response.text}")

        else:
            logger.info("Not Required... Skipping prediction.")

    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

###############################################################################################################################################################################
def unseen_data_preparation(newdata: pd.DataFrame, scale_method: str = 'standard', date_columns: list = None,
                             corr_threshold: float = 0.90, drop_corr: bool = False, date_transformation: bool = True) -> pd.DataFrame:
    """
    Prepares new data for modeling by scaling numeric features, handling date features,
    and optionally dropping highly correlated features.

    Parameters:
    - newdata: DataFrame containing the new data to be prepared.
    - scale_method: Method for scaling features (currently only 'standard' is supported).
    - date_columns: List of columns to be converted into date features.
    - corr_threshold: Correlation threshold for dropping correlated features.
    - drop_corr: Whether to drop highly correlated features.
    - date_transformation: Whether to create additional date features.

    Returns:
    - A DataFrame with prepared features ready for modeling.
    """

    # Select numeric columns
    numeric_dataframe = newdata.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix and identify columns to drop based on threshold
    corr_matrix = numeric_dataframe.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    if drop_corr:
        newdata.drop(to_drop, axis=1, inplace=True)
        logger.info(f"Dropped correlated columns: {to_drop}")

    # Prepare for date transformations
    final_date = pd.DataFrame()

    if date_transformation and date_columns is not None:
        features = ["day", "month", "year", "weekday"]
        date_data = newdata[date_columns].copy()

        # Convert date columns to datetime and extract features
        for eachcol in date_data:
            date_data[eachcol] = pd.to_datetime(date_data[eachcol], errors='coerce')
            for eachfeature in features:
                col_name = f"{eachcol}_{eachfeature}"
                if eachfeature == 'day':
                    date_data[col_name] = date_data[eachcol].dt.day
                elif eachfeature == 'month':
                    date_data[col_name] = date_data[eachcol].dt.month
                elif eachfeature == 'year':
                    date_data[col_name] = date_data[eachcol].dt.year
                elif eachfeature == 'weekday':
                    date_data[col_name] = date_data[eachcol].dt.weekday
            
        date_data.drop(date_columns, axis=1, inplace=True)
        final_date = pd.get_dummies(date_data, drop_first=True, dtype='int16')

    # Scale numeric features
    column_to_scale = newdata.select_dtypes(include=['float64', 'int64', 'float32']).columns.drop('is_churn')
    
    # Standard scaling
    if scale_method == 'standard':
        transformer = StandardScaler()
        scaled_data = pd.DataFrame(transformer.fit_transform(newdata[column_to_scale]), columns=column_to_scale)
    else:
        raise ValueError(f"Unsupported scale method: {scale_method}. Only 'standard' is supported.")

    # Combine scaled data with transformed date features and target variable
    if date_transformation:
        final_df = pd.concat([scaled_data, final_date, newdata['is_churn']], axis=1)
    else:
        final_df = pd.concat([scaled_data, newdata['is_churn']], axis=1)

    # Drop unnecessary columns
    newdata = final_df.drop(columns=['is_churn', 'city', 'registered_via', 'registration_duration'], errors='ignore')
    
    logger.info("Data preparation complete.")
    return newdata

###############################################################################################################################################################################
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def view_final_predictions(db_path: str, db_file_name: str) -> pd.DataFrame:
    """
    Retrieve final predictions from the SQLite database.

    Parameters:
    - db_path: The path to the database.
    - db_file_name: The name of the database file.

    Returns:
    - A DataFrame containing the final predictions or None if an error occurs.
    """
    query = 'SELECT * FROM predictions'
    
    try:
        # Use a context manager to connect to the SQLite database
        with sqlite3.connect(db_path + db_file_name) as cnx:
            # Execute the query and return the result as a DataFrame
            final_predictions_df = pd.read_sql(query, cnx)
        
        logger.info(f"Retrieved {len(final_predictions_df)} predictions from the database.")
        return final_predictions_df

    except sqlite3.Error as e:
        logger.error(f"SQLite error retrieving Final_Predictions table: {e}")
    except Exception as e:
        logger.error(f"Error retrieving Final_Predictions table: {e}")

    return None

    
###############################################################################################################################################################################
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_change(current: float, previous: float) -> float:
    """
    Calculate the percentage change between the current and previous values.

    Parameters:
    - current: The current value.
    - previous: The previous value.

    Returns:
    - The percentage change between the current and previous values.
      Returns 0 if values are equal, 'inf' if previous is zero,
      and 0 for invalid types.
    """
    if current == previous:
        logger.info("Current and previous values are equal, returning 0.")
        return 0.0
    
    try:
        change = (abs(current - previous) / previous) * 100.0
        logger.info(f"Calculated change: {change:.2f}% between current: {current} and previous: {previous}.")
        return change
    except ZeroDivisionError:
        logger.warning("Previous value is zero, returning infinity for percentage change.")
        return float('inf')
    except TypeError:
        logger.error("Invalid types for current or previous values, returning 0.")
        return 0.0

    
###############################################################################################################################################################################   
def get_reset_process_flags() -> Dict[str, int]:
    """
    Resets the process flags to their initial state.

    Returns:
        A dictionary with process flag names as keys and their initial values set to 0.
    """
    return {
        'load_data': 0,
        'process_transactions': 0,
        'process_members': 0,
        'process_userlogs': 0,
        'merge_data': 0,
        'Data_Preparation': 0,
        'Model_Training_plain': 0,
        'Model_Training_hpTunning': 0,
        'Prediction': 0
    }


###############################################################################################################################################################################
from typing import Dict

def get_reset_process_flags_flip() -> Dict[str, int]:
    """
    Resets the process flags to their 'active' state.

    Returns:
        A dictionary with process flag names as keys and their active values set to 1.
    """
    return {
        'load_data': 1,
        'process_transactions': 1,
        'process_members': 1,
        'process_userlogs': 1,
        'merge_data': 1,
        'Data_Preparation': 1,
        'Model_Training_plain': 1,
        'Model_Training_hpTunning': 1,
        'Prediction': 1
    }


###############################################################################################################################################################################
def get_flush_db_process_flags(db_path: str, drfit_db_name: str, flip: Optional[bool] = True) -> None:
    """
    Flushes the process flags in the specified SQLite database.

    Parameters:
        db_path (str): The path to the database.
        drift_db_name (str): The name of the drift database.
        flip (Optional[bool]): If True, resets process flags to active (1), otherwise resets to inactive (0).
    """
    try:
        # Establish a connection to the SQLite database
        with sqlite3.connect(db_path + drfit_db_name) as cnx:
            # Get the appropriate process flags based on the flip parameter
            process_flags = get_reset_process_flags_flip() if flip else get_reset_process_flags()
            
            # Create a DataFrame from the process flags dictionary
            process_flags_df = pd.DataFrame(process_flags, index=[0])
            
            # Write the DataFrame to the process_flags table in the database
            process_flags_df.to_sql(name='process_flags', con=cnx, if_exists='replace', index=False)
            logger.info("Process flags flushed successfully.")

    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


###############################################################################################################################################################################
def get_difference(df):
    """
    Calculates the percentage change between the 'new' and 'old' columns of the DataFrame.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the 'new' and 'old' columns.

    Returns:
        pd.Series: A Series containing the percentage change for each row.
    """
    # Ensure that 'new' and 'old' columns exist in the DataFrame
    if 'new' in df.columns and 'old' in df.columns:
        # Calculate the percentage change for each row
        percent_change = df.apply(lambda row: get_change(row['new'], row['old']), axis=1)
        return percent_change
    else:
        raise ValueError("DataFrame must contain 'new' and 'old' columns.")


###############################################################################################################################################################################    
def get_data_drift(current_data, old_data, column_list, exclude_list, cnx, metric='std'):
    """
    Calculate and save data drift metrics (standard deviation or mean) between current and old datasets.

    Parameters:
        current_data (pd.DataFrame): The current dataset.
        old_data (pd.DataFrame): The old dataset.
        column_list (list): List of columns to check for drift.
        exclude_list (list): List of columns to exclude from the drift check.
        cnx (sqlite3.Connection): SQLite database connection for saving the results.
        metric (str): The metric to use for drift calculation ('std' for standard deviation or 'mean' for mean).

    Returns:
        float: Mean of the percentage differences for the chosen metric.
    """
    drift_dict = {'old': {}, 'new': {}}
    deviation_percentage = []

    for each_col in column_list:
        if each_col not in exclude_list:
            if metric == 'std':
                std_current = current_data[each_col].std()
                std_old = old_data[each_col].std()
                drift_dict['new'][each_col] = std_current
                drift_dict['old'][each_col] = std_old
                deviation_percentage.append(get_change(std_current, std_old))
            elif metric == 'mean':
                mean_current = current_data[each_col].mean()
                mean_old = old_data[each_col].mean()
                drift_dict['new'][each_col] = mean_current
                drift_dict['old'][each_col] = mean_old
                deviation_percentage.append(get_change(mean_current, mean_old))

    # Drift Dict Saving
    print(drift_dict)
    df = pd.DataFrame(drift_dict)
    df['prcnt_difference'] = df.apply(get_difference, axis=1) 
    df['column_name'] = df.index
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['time'] = timestamp
    print(df)
    df.to_sql(name='drift_df', con=cnx, if_exists='append', index=False)

    return np.mean(deviation_percentage) if deviation_percentage else float('nan')

    
###########################################################################################################################################################################
def get_drift(old_data_directory: str, new_data_directory: str, db_path: str, drift_db_name: str,
              metric: str = 'std', start_date: str = '2017-03-01', end_date: str = '2017-03-31', 
              chunk_size: int = 50000) -> None:
    """
    Calculate data drift between new and old datasets and save results to an SQLite database.

    Parameters:
        old_data_directory (str): Path to the directory containing old data files.
        new_data_directory (str): Path to the directory containing new data files.
        db_path (str): Path to the SQLite database.
        drift_db_name (str): Name of the SQLite database for drift results.
        metric (str): Metric to calculate drift ('std' or 'mean').
        start_date (str): Start date for filtering new data.
        end_date (str): End date for filtering new data.
        chunk_size (int): Number of rows to read per chunk from CSV files.
    """
    
    # Establish a connection to the SQLite database using a context manager
    with sqlite3.connect(db_path + drift_db_name) as cnx:
        try:
            # Load new data using chunk processing
            march_user_logs = load_and_filter_data(f"{new_data_directory}user_logs_new.csv", 
                                                    'date', start_date, end_date, chunk_size)
            march_transactions = load_and_filter_data(f"{new_data_directory}transactions_logs_new.csv", 
                                                       'transaction_date', start_date, end_date, chunk_size)

            # Compress new dataframes to reduce memory usage
            march_user_logs, _, _ = compress_dataframes([march_user_logs])[0]
            march_transactions, _, _ = compress_dataframes([march_transactions])[0]

            # Load old data in chunks and compress
            transactions = load_and_concatenate_chunks(f"{old_data_directory}transactions_logs.csv", chunk_size)
            user_logs = load_and_concatenate_chunks(f"{old_data_directory}userlogs.csv", chunk_size)

            # Apply time fixes to the necessary columns
            transactions['transaction_date'] = fix_time_in_df(transactions, 'transaction_date', expand=False)
            transactions['membership_expire_date'] = fix_time_in_df(transactions, 'membership_expire_date', expand=False)
            user_logs['date'] = fix_time_in_df(user_logs, 'date', expand=False)

            # Compress old dataframes to reduce memory usage
            transactions, _, _ = compress_dataframes([transactions])[0]
            user_logs, _, _ = compress_dataframes([user_logs])[0]

            # Select columns for drift analysis (only numerical)
            column_list_tran = transactions.select_dtypes(include=['int8', 'int16', 'int32', 'float16']).columns.tolist()
            column_list_userlogs = user_logs.select_dtypes(include=['int8', 'int16', 'int32', 'float16']).columns.tolist()

            # Define columns to exclude from drift calculation
            exclude_list_tran = ['date'] 
            exclude_list_user_log = ['transaction_date', 'membership_expire_date']

            # Compute data drift
            user_logs_drift = get_data_drift(march_user_logs, user_logs, column_list_userlogs, exclude_list_user_log, cnx, metric=metric)
            transactions_drift = get_data_drift(march_transactions, transactions, column_list_tran, exclude_list_tran, cnx, metric=metric)

            # Display drift results
            print(f"User Logs Data Drift ({metric}): {user_logs_drift}")
            print(f"Transaction Data Drift ({metric}): {transactions_drift}")

            # Create a DataFrame to store the drift results
            drift = pd.DataFrame({
                'drift_userlog': [user_logs_drift],
                'drift_transaction': [transactions_drift]
            })

            # Save drift results to the SQLite database
            drift.to_sql(name='drift', con=cnx, if_exists='replace', index=False)
            print(f"Writing to Database Done at {db_path + drift_db_name}")

            # Trigger any post-processing actions
            get_drift_trigger(db_path, drift_db_name)

        except Exception as e:
            print(f"An error occurred: {e}")

def load_and_filter_data(file_path: str, date_column: str, start_date: str, end_date: str, chunk_size: int) -> pd.DataFrame:
    """
    Load and filter data from a CSV file by date range.

    Parameters:
        file_path (str): Path to the CSV file.
        date_column (str): Column name containing the date information.
        start_date (str): Start date for filtering.
        end_date (str): End date for filtering.
        chunk_size (int): Number of rows to read per chunk.

    Returns:
        pd.DataFrame: Filtered DataFrame containing the data.
    """
    data_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk[date_column] = pd.to_datetime(chunk[date_column], errors='coerce')
        filtered_chunk = chunk[(chunk[date_column] >= start_date) & (chunk[date_column] <= end_date)]
        data_chunks.append(filtered_chunk)

    return pd.concat(data_chunks, ignore_index=True)

def load_and_concatenate_chunks(file_path: str, chunk_size: int) -> pd.DataFrame:
    """
    Load and concatenate data from a CSV file in chunks.

    Parameters:
        file_path (str): Path to the CSV file.
        chunk_size (int): Number of rows to read per chunk.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing the data.
    """
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)



###############################################################################################################################################################################
def get_drift_trigger(db_path, drfit_db_name):
    """
    Triggers post-processing actions based on the calculated data drift value.
    
    This function connects to an SQLite database to retrieve the drift results,
    calculates the mean drift value, and sets various process flags based on the
    drift value thresholds. The flags indicate whether to proceed with data loading,
    processing, or model retraining actions. The process flags are stored back in
    the database for further processing.

    Parameters:
    ----------
    db_path : str
        The path to the directory containing the database file.
    drift_db_name : str
        The name of the database file containing drift results.

    Returns:
    -------
    None
    """
    try:
        cnx = sqlite3.connect(db_path + drfit_db_name)
        
        process_flags = get_reset_process_flags()
        
        print("Before Change process_flags", process_flags) 

        drift = pd.read_sql('SELECT * FROM drift', cnx)
        drift_value = drift.mean(axis=1)[0]
        print("Drift_value.......", drift_value) 
        
        # Define drift value thresholds for processing actions
        if 0 <= drift_value <= 10:
            print("No Change since drift is low")
            
        elif 10 < drift_value <= 20:
            process_flags['load_data'] = 1
            process_flags['process_transactions'] = 1
            process_flags['process_members'] = 1
            process_flags['process_userlogs'] = 1
            process_flags['merge_data'] = 1
            process_flags['Data_Preparation'] = 1
            process_flags['Model_Training_plain'] = 1
            
        elif 20 < drift_value <= 30:
            process_flags['load_data'] = 1
            process_flags['process_transactions'] = 1
            process_flags['process_members'] = 1
            process_flags['process_userlogs'] = 1
            process_flags['merge_data'] = 1
            process_flags['Data_Preparation'] = 1
            process_flags['Model_Training_hpTunning'] = 1
            
        else:
            print("Drift is very high, please re-run the development notebook.")
            
        print("After Change process_flags", process_flags) 
        process_flags_df = pd.DataFrame(process_flags, index=[0])
        process_flags_df.to_sql(name='process_flags', con=cnx, if_exists='replace', index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cnx' in locals():
            cnx.close()  # Ensure the database connection is closed

        
###############################################################################################################################################################################        
        
        
        
        
    
    