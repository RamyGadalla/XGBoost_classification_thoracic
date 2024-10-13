# Thoracic Surgery Risk of Survival - Classification with XGBoost

## Overview
This repository contains a Python code for classification of the **Thoracic Surgery Risk of Survival** dataset using the **XGBoost** machine learning algorithm. The objective is to predict the one-year survival status of patients who underwent thoracic surgery.

The dataset provides medical and demographic information about patients, and the classification task aims to assist in identifying risk factors and predicting patient outcomes, helping healthcare providers make informed decisions.

## Files
- **thoracic_surgery_classification.py**: The main Python script that loads the dataset, preprocesses the data, trains an XGBoost model, and evaluates the classification performance.

## Requirements
To run the script, you will need the following Python packages:
- **xgboost**
- **pandas**
- **scikit-learn**
- **numpy**



## Usage
1. Clone the repository:
```sh
git clone https://github.com/yourusername/thoracic_surgery_classification.git
```

## Dataset
The **Thoracic Surgery Risk of Survival** dataset includes various features such as age, smoking history, and other clinical details. The target variable is the **one-year survival status** of the patients.


### Data Caveats
- The dataset size is relatively small, which may limit the generalizability of the model.
- The dataset is based on medical records from a specific group of patients, which may introduce bias and affect the model's performance on different populations.
- The dataset may have **class imbalance**, with one class (e.g., survivors) being significantly more frequent than the other. This can impact model performance and requires appropriate handling, such as resampling techniques or the use of class weights.


## Model
- The **XGBoost** algorithm is used to perform classification, leveraging gradient boosting for improved accuracy and performance.

## Results
The script provides the following model evaluation metrics:
- **Accuracy**
- **Precision**
- **Confusion Matrix**
- **Features Importance**

These metrics help assess the performance of the model in predicting one-year survival status.


## Acknowledgements
- Dataset source: [Thoracic Surgery Risk of Survival Dataset](https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data)
