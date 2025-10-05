# Diabetes Prediction Using 2014 BRFSS Dataset

## Project Overview
This repository contains the code and documentation for a data science class project (ANA500) aimed at predicting diabetes occurrence using a sub-sample of the 2014 Behavioral Risk Factor Surveillance System (BRFSS) dataset. The goal is to preprocess the dataset, perform exploratory data analysis (EDA), and develop neural network and deep learning models to predict diabetes status. The target variable, `DIABETE3`, is recoded into a binary variable (`DIABETE3_binary`: 1 = Yes [diagnosed diabetes or pre-diabetes], 0 = No [gestational or no diabetes]) to simplify classification for neural network modeling in Modules 3/4. The project includes extensive data cleaning, imputation of missing values and non-response codes, and preparation for advanced modeling.

### Objectives
- **Data Preprocessing**: Clean the BRFSS dataset by handling missing values, non-response codes (e.g., 7, 9, 77, 99), and converting `WEIGHT2` from kilograms to pounds.
- **Exploratory Data Analysis (EDA)**: Analyze relationships between diabetes status (`DIABETE3_binary`) and predictors like `_BMI5CAT`, `WEIGHT2`, and `GENHLTH` using visualizations (e.g., bar plots, histograms).
- **Modeling**: Build neural network and deep learning models to predict diabetes occurrence, addressing class imbalance with class weights.
- **Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, and AUC for binary classification.

## Dataset
The dataset is a subset of the 2014 BRFSS survey with 5,000 rows and 34 variables, all encoded as `int64` or `float64`. Key variables include:
- **Target**: `DIABETE3` (diabetes status, recoded to `DIABETE3_binary`: 1 = Yes [categories 1, 4], 0 = No [categories 2, 3]).
- **Predictors**: `_BMI5CAT` (BMI categories), `WEIGHT2` (weight in pounds), `GENHLTH` (general health), `INCOME2` (income level), and others.
- **Missing Values and Codes**: Variables have missing (`NaN`) values and non-response codes (e.g., 7, 9, 77, 99) indicating "Don't know" or "Refused," imputed using mode, median, or KNN methods.

### Data Preprocessing Steps
1. **WEIGHT2 Conversion**:
   - Converted weights from kilograms (9000–9998) to pounds using 1 kg ≈ 2.20462 lbs.
   - Kept pounds (50–999) unchanged.
   - Set invalid codes (7777, 9999) to `NaN` for imputation.
2. **Mode Imputation** (14 categorical/binary variables):
   - Variables: `EMPLOY1`, `MARITAL`, `_RACE`, `CHILDREN`, `CVDCRHD4`, `CHCKIDNY`, `_TOTINDA`, `ADDEPEV2`, `EXERANY2`, `HLTHPLN1`, `_STATE`, `ASTHMA3`, `DIABETE3`, `RENTHOM1`.
   - Set non-response codes (e.g., 7, 9, 66, 99) to `NaN` and imputed with mode.
3. **Median Imputation** (7 ordinal/count-like variables):
   - Variables: `INCOME2`, `GENHLTH`, `_AGEG5YR`, `_EDUCAG`, `CHECKUP1`, `SLEPTIM1`, `MENTHLTH`.
   - Set codes (e.g., 77, 99, 14) to `NaN` and imputed with median.
   - Recoded `MENTHLTH` 88 to 0 (no days) before imputation.
4. **KNN Imputation** (10 variables with high missing or correlations):
   - Variables: `MSCODE`, `HLTHCVR1`, `NUMADULT`, `DRVISITS`, `FLUSHOT6`, `DECIDE`, `BLIND`, `USEEQUIP`, `_BMI5CAT`, `WEIGHT2`.
   - Set codes (e.g., 7, 9, 77, 9999) to `NaN`, imputed using KNN (n_neighbors=5), and mapped categorical variables to valid values (e.g., 1, 2 for `DECIDE`).
   - `DRVISITS` recoded 88 to 0 before imputation.
5. **DIABETE3 Recoding**:
   - Pre-imputation distribution: 1 (627), 2 (39), 3 (4251), 4 (76), 7 (2), 9 (5).
   - Post-mode imputation: 1 (627), 2 (39), 3 (4258), 4 (76).
   - Recoded to `DIABETE3_binary`: 1 and 4 = Yes (703), 2 and 3 = No (4297).
   - Original `DIABETE3` retained for EDA.

## Repository Structure
