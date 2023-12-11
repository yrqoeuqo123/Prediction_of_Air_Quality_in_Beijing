# Prediction of Air Quality in Beijing

## Overview
This repository hosts the code and results for a study on air pollution in Beijing, using data from March 2013 to February 2017. It focuses on predicting PM2.5 and PM10 levels, leveraging machine learning techniques and data from 12 monitoring stations.

## Repository Structure

### Data Folder
- `raw_data`: Original datasets from various sources.
- `preprocessed_data`: Datasets after initial preprocessing (cleaning, normalization).
- `train_test_split_data`: Data split into training and testing sets.

### Scripts Folder
- `pearson_corr_heatmap.py`: Generates Pearson correlation heatmaps.
- `find_optimal_2.5and10.py`: Identifies optimal machine learning models for PM2.5 and PM10 prediction, including hyperparameter tuning.
- `train_with_optimal_2.5and10.py`: Trains models using optimal parameters found by `find_optimal_2.5and10.py`.
- `train_test_split.py`: Splits the dataset into training and testing sets.
- `preprocessing.py`: Conducts data preprocessing like cleaning and encoding.

## How to Use This Repository
1. **Setting Up**: Clone the repository and install the required Python libraries.
2. **Data Preparation**: Use `preprocessing.py` for data cleaning and normalization. Then, use `train_test_split.py` to split the data for model training.
3. **Model Selection and Training**: Run `find_optimal_2.5and10.py` to determine the best model parameters, and then train the models using `train_with_optimal_2.5and10.py`.
4. **Analysis**: Generate correlation heatmaps with `pearson_corr_heatmap.py`.
5. **Results**: Check output folders for datasets, predictions, visualizations, and performance metrics.

## Replicability
This project is designed for full replicability. Follow the outlined steps using the provided scripts for a comprehensive study replication.

## Required Python Libraries (Minimum Version Required)
- [Numpy v1.21.0](https://numpy.org/)
- [Pandas v2.0.3](https://pandas.pydata.org/)
- [scikit-learn v1.3.2](https://scikit-learn.org/stable/)
- [matplotlib v3.8.2](https://matplotlib.org/)
- [seaborn v0.12.0b3](https://seaborn.pydata.org/)
- [calendar v2.5](https://docs.python.org/3/library/calendar.html)
- [math v3.8](https://docs.python.org/3/library/math.html)
- [tqdm v2.2.3](https://tqdm.github.io/)
- [tabulate v0.8.0](https://pypi.org/project/tabulate/)

## Contributors
This project was developed by a team of dedicated students from Emory University. We are passionate about using data science to address environmental issues and improve public health outcomes. Our team combines expertise in machine learning, data analysis, and environmental science to provide insightful research on air quality in Beijing.

