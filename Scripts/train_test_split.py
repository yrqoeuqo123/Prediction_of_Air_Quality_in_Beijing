import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_heatmap(corr_matrix, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()

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

        # Concatenate the scaled training and test data
        combined_scaled_df = pd.concat([
            pd.DataFrame(train_scaled, columns=features, index=train.index),
            pd.DataFrame(test_scaled, columns=features, index=test.index)
        ])

        # Concatenate the target variables
        combined_targets = pd.concat([train[['PM2.5', 'PM10']], test[['PM2.5', 'PM10']]])

        # Combine scaled features with target variables
        combined_df = pd.concat([combined_scaled_df, combined_targets], axis=1)

        # Calculate the Pearson correlation matrix
        corr_matrix_pm25 = combined_df.corrwith(combined_df['PM2.5']).to_frame().T
        corr_matrix_pm10 = combined_df.corrwith(combined_df['PM10']).to_frame().T

        # Plot heatmaps for PM2.5 and PM10 correlations
        plot_heatmap(corr_matrix_pm25, f'Pearson Correlation with PM2.5 - {file_name}')
        plot_heatmap(corr_matrix_pm10, f'Pearson Correlation with PM10 - {file_name}')



        pd.DataFrame(train_scaled, columns=features).to_csv(f"{file_name[:-4]}_xTrain.csv", index=False)
        pd.DataFrame(test_scaled, columns=features).to_csv(f"{file_name[:-4]}_xTest.csv", index=False)

if __name__ == "__main__":
    main()
