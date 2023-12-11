import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_lag_features(df, lag_days=1):
    """
    Create lag features for time series data.

    Parameters:
    df (DataFrame): The input data frame containing time series data.
    lag_days (int): The number of lag days to create features for.

    Returns:
    DataFrame: The input data frame augmented with lag features.
    """
    
    for lag in range(1, lag_days + 1):
        df[f'PM2.5_lag_{lag}'] = df['PM2.5'].shift(lag)
        df[f'PM10_lag_{lag}'] = df['PM10'].shift(lag)
    df.dropna(inplace=True)  # Drop rows with NaN values created by shifting
    return df

def main():
    """
    Main function to process multiple CSV files containing air quality data.

    This function reads each file, creates lag features, performs a time-based
    train-test split, scales the features, and saves the processed data.
    """
    file_names = [
        "Aotizhongxin_air_quality_20130301-20170228_preprocessed.csv",
        "Changping_air_quality_20130301-20170228_preprocessed.csv",
        "Dingling_air_quality_20130301-20170228_preprocessed.csv",
        "Dongsi_air_quality_20130301-20170228_preprocessed.csv",
        "Guanyuan_air_quality_20130301-20170228_preprocessed.csv",
        "Gucheng_air_quality_20130301-20170228_preprocessed.csv",
        "Huairou_air_quality_20130301-20170228_preprocessed.csv",
        "Nongzhanguan_air_quality_20130301-20170228_preprocessed.csv",
        "Shunyi_air_quality_20130301-20170228_preprocessed.csv",
        "Tiantan_air_quality_20130301-20170228_preprocessed.csv",
        "Wanliu_air_quality_20130301-20170228_preprocessed.csv",
        "Wanshouxigong_air_quality_20130301-20170228_preprocessed.csv"
    ]


    for file_name in file_names:
        df = pd.read_csv(file_name, encoding='ISO-8859-1')

        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Create lag features
        df = create_lag_features(df, lag_days=1)

        # Determine the 80% mark for the split
        total_rows = len(df)
        split_row = int(total_rows * 0.8)

        # Split the data based on calculated row index
        train = df.iloc[:split_row]
        test = df.iloc[split_row:]

        # Preprocess features (scaling)
        scaler = StandardScaler()
        features = train.columns.drop(['PM2.5', 'PM10', 'date'])
        train_scaled = scaler.fit_transform(train[features])
        test_scaled = scaler.transform(test[features])

        # Save the processed datasets
        train[['PM2.5', 'PM10']].to_csv(f"{file_name[:-4]}_yTrain.csv", index=False)
        test[['PM2.5', 'PM10']].to_csv(f"{file_name[:-4]}_yTest.csv", index=False)
        pd.DataFrame(train_scaled, columns=features).to_csv(f"{file_name[:-4]}_xTrain.csv", index=False)
        pd.DataFrame(test_scaled, columns=features).to_csv(f"{file_name[:-4]}_xTest.csv", index=False)

if __name__ == "__main__":
    main()
