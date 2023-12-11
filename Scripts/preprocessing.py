import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from tqdm import tqdm

'''
This script is designed to preprocess and standardize air quality data from multiple CSV files.
The preprocessing steps include dropping unnecessary columns, imputing missing values, 
one-hot encoding categorical features,calculating holiday proximity, 
and adding additional data-related features.

'''

# List of public holidays in China from 2013 to 2017 (YYYY-MM-DD format)
holidays = [
    # 2013 Holidays
    "2013-01-01", "2013-01-02", "2013-01-03", "2013-02-09", "2013-02-10",
    "2013-02-11", "2013-02-12", "2013-02-13", "2013-02-14", "2013-02-15",
    "2013-02-24", "2013-03-08", "2013-03-12", "2013-03-13", "2013-03-20",
    "2013-04-04", "2013-04-05", "2013-04-06", "2013-04-29", "2013-04-30",
    "2013-05-01", "2013-06-10", "2013-06-11", "2013-06-12", "2013-09-19",
    "2013-09-20", "2013-09-21", "2013-10-01", "2013-10-02", "2013-10-03",
    "2013-10-04", "2013-10-05", "2013-10-06", "2013-10-07", "2013-11-08",
    "2013-12-22", "2013-12-25",

    # 2014 Holidays
    "2014-01-01", "2014-01-26", "2014-01-30", "2014-01-31", "2014-02-01",
    "2014-02-02", "2014-02-03", "2014-02-04", "2014-02-05", "2014-02-06",
    "2014-02-08", "2014-02-14", "2014-03-02", "2014-03-08", "2014-03-12",
    "2014-03-21", "2014-04-05", "2014-04-06", "2014-04-07", "2014-05-01",
    "2014-05-02", "2014-05-03", "2014-05-04", "2014-05-31", "2014-06-01",
    "2014-06-02", "2014-06-21", "2014-07-01", "2014-07-11", "2014-08-01",
    "2014-08-02", "2014-08-10", "2014-09-03", "2014-09-06", "2014-09-07",
    "2014-09-08", "2014-09-10", "2014-09-23", "2014-09-28", "2014-10-01",
    "2014-10-02", "2014-10-03", "2014-10-04", "2014-10-05", "2014-10-06",
    "2014-10-07", "2014-10-11", "2014-11-07", "2014-11-08", "2014-11-09",
    "2014-11-10", "2014-11-11", "2014-11-12", "2014-12-22", "2014-12-25",

    # 2015 Holidays
    "2015-06-20", "2015-06-21", "2015-06-22", "2015-07-01", "2015-07-11",
    "2015-08-01", "2015-08-20", "2015-08-28", "2015-09-03", "2015-09-10",
    "2015-09-23", "2015-09-27", "2015-10-01", "2015-10-02", "2015-10-03",
    "2015-10-04", "2015-10-05", "2015-10-06", "2015-10-07", "2015-10-10",
    "2015-10-21", "2015-11-08", "2015-12-22", "2015-12-25",

    # 2016 Holidays
    "2016-01-01", "2016-01-02", "2016-01-03", "2016-02-07", "2016-02-08",
    "2016-02-09", "2016-02-10", "2016-02-11", "2016-02-12", "2016-02-13",
    "2016-02-22", "2016-03-08", "2016-03-10", "2016-03-12", "2016-03-20",
    "2016-04-03", "2016-04-04", "2016-04-30", "2016-05-01", "2016-05-02",
    "2016-05-04", "2016-06-01", "2016-06-09", "2016-06-10", "2016-06-11",
    "2016-06-21", "2016-07-01", "2016-07-11", "2016-08-01", "2016-08-09",
    "2016-08-17", "2016-09-10", "2016-09-15", "2016-09-16", "2016-09-17",
    "2016-09-18", "2016-09-22", "2016-10-01", "2016-10-02", "2016-10-03",
    "2016-10-04", "2016-10-05", "2016-10-06", "2016-10-07", "2016-10-08",
    "2016-10-09", "2016-10-09", "2016-11-08", "2016-12-21", "2016-12-25",

    # 2017 Holidays
    "2017-01-01", "2017-01-02", "2017-01-22", "2017-01-27", "2017-01-28",
    "2017-01-29", "2017-01-30", "2017-01-31", "2017-02-01", "2017-02-02",
    "2017-02-04", "2017-02-11", "2017-02-27", "2017-03-08", "2017-03-12",
    "2017-03-20", "2017-04-01", "2017-04-02", "2017-04-03", "2017-04-04",
    "2017-04-29", "2017-04-30", "2017-05-01", "2017-05-04", "2017-05-27",
    "2017-05-28", "2017-05-29", "2017-05-30", "2017-06-01", "2017-06-21",
    "2017-07-01", "2017-07-11", "2017-08-01", "2017-08-28", "2017-09-05",
    "2017-09-10", "2017-09-23", "2017-09-30", "2017-10-01", "2017-10-02",
    "2017-10-03", "2017-10-04", "2017-10-05", "2017-10-06", "2017-10-07",
    "2017-10-08", "2017-10-28", "2017-11-08", "2017-12-22", "2017-12-25"
]

# Precompute holiday proximity values and store them in a dictionary
holiday_proximity_dict = {}
for date in pd.date_range(start="2013-01-01", end="2017-12-31"):
    nearest_holiday = min(holidays, key=lambda x: abs(
        (date - pd.to_datetime(x)).days))
    days_until_holiday = (pd.to_datetime(nearest_holiday) - date).days
    holiday_proximity_dict[date] = days_until_holiday

# Function to calculate holiday proximity using the precomputed dictionary

def calculate_holiday_proximity(date):
    """
    Calculate holiday proximity for a given date.

    Parameters:
    date : pandas.Timestamp
        The date for which holiday proximity is calculated.

    Returns:
    int
        Number of days until the nearest holiday.
    """

    return holiday_proximity_dict[date]

# Function to standardize numerical features


def standardize_numerical_features(df):
    """
    Standardize numerical features in a DataFrame.

    Parameters:
    df : pandas.DataFrame
        Input DataFrame containing numerical features.

    Returns:
    pandas.DataFrame
        DataFrame with standardized numerical features.
    """

    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# Function to preprocess the data


def preprocess(df):
    """
    Preprocess the air quality data.

    Parameters:
    df : pandas.DataFrame
        Input DataFrame containing raw air quality data.

    Returns:
    pandas.DataFrame
        Preprocessed DataFrame with added features.
    """

    # Remove the 'No' column
    df.drop('No', axis=1, inplace=True)

    # Combine year, month, day into a single date column and drop them
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(
        df.select_dtypes(include=[np.number]))

    # Standardize numerical features
    df = standardize_numerical_features(df)

    # One-hot encode 'wd' and 'station'
    encoder = OneHotEncoder(sparse_output=False)
    for col in ['wd', 'station']:
        encoded = encoder.fit_transform(df[[col]])
        df = df.drop(col, axis=1)
        encoded_df = pd.DataFrame(
            encoded, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]])
        df = pd.concat([df, encoded_df], axis=1).reset_index(drop=True)

    # Calculate holiday proximity
    df['holiday_proximity'] = df['date'].apply(calculate_holiday_proximity)

    # Add a column 'is_holiday' to indicate whether the date is a holiday
    df['is_holiday'] = df['date'].isin(holidays).astype(int)

    # Add additional data-related features (example: day of the week, etc.)
    df['day_of_week'] = df['date'].dt.dayofweek
    # Add more features as needed

    return df

# Load and preprocess each CSV file


def main():
    """
    Main function to process multiple CSV files containing air quality data.
    """

    file_names = [
        "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
        "PRSA_Data_Changping_20130301-20170228.csv",
        "PRSA_Data_Dingling_20130301-20170228.csv",
        "PRSA_Data_Dongsi_20130301-20170228.csv",
        "PRSA_Data_Guanyuan_20130301-20170228.csv",
        "PRSA_Data_Gucheng_20130301-20170228.csv",
        "PRSA_Data_Huairou_20130301-20170228.csv",
        "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
        "PRSA_Data_Shunyi_20130301-20170228.csv",
        "PRSA_Data_Tiantan_20130301-20170228.csv",
        "PRSA_Data_Wanliu_20130301-20170228.csv",
        "PRSA_Data_Wanshouxigong_20130301-20170228.csv"
    ]

    for file_name in tqdm(file_names, desc="Processing Files"):
        df = pd.read_csv(file_name, encoding='ISO-8859-1')
        # Extract the station name from the file name
        station_name = file_name.split('_')[2]
        # Manually add the station column
        df['station'] = station_name
        preprocessed_df = preprocess(df)

        # Extract year range from the file name
        year_range = file_name.split('_')[3].split('.')[0]

        # Create a descriptive file name
        descriptive_file_name = f"{station_name}_air_quality_{year_range}_preprocessed.csv"

        # Save the preprocessed data with the descriptive file name
        preprocessed_df.to_csv(descriptive_file_name, index=False)


if __name__ == "__main__":
    main()
