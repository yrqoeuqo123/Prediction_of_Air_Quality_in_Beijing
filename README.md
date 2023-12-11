# Prediction_of_Air_Quality_in_Beijing
Investigating Beijing's air quality, this study uses the "Beijing Multi-Site Air-Quality Data" (March 2013 - February 2017) to predict PM2.5 and PM10 levels. Emphasizing machine learning model evaluation, it analyzes pollution patterns from 12 monitoring stations, assessing urban air pollution dynamics.

## Overview
This repository contains all the necessary code and results for the study on air pollution in Beijing. It's structured to ensure replicability and ease of use. The project is divided into two main directories: data and scripts, each serving a specific purpose in the research workflow.

## Repository Structure
### Data Folder
The data directory is organized into subfolders to manage different stages of data processing:

- raw_data: Contains the original datasets as obtained from the sources.
- preprocessed_data: Stores the datasets after initial preprocessing steps like cleaning and normalization.
- train_test_split_data: Holds the data split into training and testing sets for model validation.

### Scripts Folder
The scripts directory houses various Python scripts, each tailored for specific tasks in the research process:

- `pearson_corr_heatmap.py`: Generates Pearson correlation heatmaps to explore relationships between different environmental variables.
- `find_optimal_2.5and10.py`: Dedicated to finding the optimal machine learning models for predicting PM2.5 and PM10 levels. It includes hyperparameter tuning and model selection.
- `train_with_optimal_2.5and10.py`: Utilizes the optimal model parameters identified by `find_optimal_2.5and10.py` to train the models on the full dataset.
- `train_test_split.py`: Handles the splitting of the dataset into training and testing sets, ensuring a robust validation process.
- `preprocessing.py`: Conducts initial data preprocessing tasks such as cleaning, encoding, and normalization.

## How to Use This Repository

1) Setting Up: Clone the repository and install the required python libraries listed in requirements.txt.

2) Data Preparation:

- The raw data is stored in Raw_Data folder.
- Run preprocessing.py to clean and normalize the data. The preprocessed dataset will automatically be saved under the same directory of the running python file.
- Execute train_test_split.py to split the data into training and testing sets. The data will be splitted into 4 files: xTrain, yTrain, xTest, yTest. x~ represent the feature set, and y~ represent target variable set. The files will be saved under the same directory of the running python file.

3) Model Selection and Training:

- Use `find_optimal_2.5and10.py` to identify the best models and their parameters.
- Train the models on your dataset using `train_with_optimal_2.5and10.py`.

4) Analysis:

- To analyze correlations between variables, run `pearson_corr_heatmap.py`.
- The output will be stored in a designated folder for results.
  
5) Results:

- Check the output folders in data for processed datasets and model predictions.

- Visualizations and models' performance metrics are saved in the results folder.

### Replicability
This project is designed for full replicability. By following the steps outlined above and using the provided scripts, you can replicate the entire study or apply the methodology to similar datasets.

### Requirement Python Libraries
Please use the version new than or equal to the python library in the list
 - Numpy v1.21.0
 - Pandas v2.0.3
 - sklearn v1.3.2
 - matplotlib v3.8.2
 - seaborn v0.12.0b3
 - calendar v2.5
 - math v3.8
 - tqdm v2.2.3
