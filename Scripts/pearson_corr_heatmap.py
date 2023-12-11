# The Python code for generating Pearson correlation heatmaps for PM2.5 and PM10 is as follows:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(corr_matrix, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title(title)
    plt.show()

def main():
    # Define file paths
    x_train_file = 'C:/Users/Tianyi Zhang/Desktop/#34/Beijing_Air_Pollution_xTrain.csv'
    y_train_25_file = 'C:/Users/Tianyi Zhang/Desktop/#34/Beijing_Air_Pollution_2.5yTrain.csv'
    y_train_10_file = 'C:/Users/Tianyi Zhang/Desktop/#34/Beijing_Air_Pollution_10yTrain.csv'
    x_test_file = 'C:/Users/Tianyi Zhang/Desktop/#34/Beijing_Air_Pollution_xTest.csv'
    y_test_25_file = 'C:/Users/Tianyi Zhang/Desktop/#34/Beijing_Air_Pollution_2.5yTest.csv'
    y_test_10_file = 'C:/Users/Tianyi Zhang/Desktop/#34/Beijing_Air_Pollution_10yTest.csv'
    
    # Load the datasets
    x_train = pd.read_csv(x_train_file)
    y_train_25 = pd.read_csv(y_train_25_file)
    y_train_10 = pd.read_csv(y_train_10_file)
    x_test = pd.read_csv(x_test_file)
    y_test_25 = pd.read_csv(y_test_25_file)
    y_test_10 = pd.read_csv(y_test_10_file)

    # Concatenate train and test sets
    x_combined = pd.concat([x_train, x_test], ignore_index=True)
    y_combined_25 = pd.concat([y_train_25, y_test_25], ignore_index=True)
    y_combined_10 = pd.concat([y_train_10, y_test_10], ignore_index=True)

    # Combine features with target variables
    combined_df_25 = pd.concat([x_combined, y_combined_25], axis=1)
    combined_df_10 = pd.concat([x_combined, y_combined_10], axis=1)

    # Calculate the Pearson correlation matrix
    corr_matrix_25 = combined_df_25.corr()
    corr_matrix_10 = combined_df_10.corr()

    # Plot heatmaps for PM2.5 and PM10 correlations
    plot_heatmap(corr_matrix_25, 'Pearson Correlation Heatmap with PM2.5')
    plot_heatmap(corr_matrix_10, 'Pearson Correlation Heatmap with PM10')

if __name__ == "__main__":
    main()
