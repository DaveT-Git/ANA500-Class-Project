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
6. **Correlation Analysis**:
   - `EXERANY2` and `_TOTINDA` had a correlation of 1, eliminated `_TOTINDA` and will use `EXERANY2`.  Both were binary yes/no categorical variables.
   - `WEIGHT2_capped` and `_BMI5CAT` had a correlation of 0.77, eliminated `WEIGHT2_capped` and will use `_BMI5CAT`. `_BMI5CAT` is a categorical variable with 4 easily understood           categories, (underweight, normal, overweight and obese).
7. **Additional Recoding**:
   - `INCOME2` was recoded from 8 narrow unbalanced categories to 5 broader and better balanced categories (`INCOME_recoded`: 1=<20k, 2=20-35k, 3=35-50k, 4=50-75k, 5=>75k).
   - `_AGEG5YR` was recoded from 13 five-year categories to 4 categories (`AGE_recoded`: 1=18-34, 2=35-49, 3=50-65, 4=65-99).
   - `_BMI5CAT` was recoded from 4 categories to three categories by combining underweight and normal into one category resulting in (`BMI_recoded`: 1=Normal, 2=Overwight, 3=Obese).
   - `DRVISITS_capped` had several floating point numbers that resulted in over 40 categories, so they were recoded into integers resulting in 12 categories, with 12 being the capped       variable from the outlier capping previously performed (`DRVISITS_recoded`).
   - `EMPLOY1` was recoded from 8 to 4 categories. ‘Employed for wages’ and ‘Self-employed’ were combined into ‘Working.’ ‘Retired’ remained its own category. ‘Homemaker’ and               ‘student’ were combined, and ‘out of work >= to 1 year,’ ‘out of work < 1 year’ and ‘unable to work’ were combined, resulting in (`EMPLOY_recoded`: 1=Working, 2=Retired,              3=Homemaker/Student, 4=Out of/unable to work).
   - `MENTHLTH`, which represented the number of days the respondent reported their mental health being “not good” over the last 30 days, which resulted in 27 categories, was recoded       into four categories (`MNTLHLTH_recoded`: 0 = 0 days, 1 = 1-5 days, 2 = 6-15 days, 3 = 16-30 days).

## Feature selection process

To determine the most relevant variables for predicting diabetes, we explored several statistical methods and blended the results with domain expertise.

### Methods
*   **Principal Component Analysis (PCA):** Analyzed feature loadings on the first five principal components, which collectively captured approximately 32.2% of the total variance. The analysis identified important variables related to health (`GENHLTH`, `MENTHLTH_recode`), demographics (`AGE_recoded`), and socioeconomic status (`INCOME_recoded`).
*   **Recursive Feature Elimination (RFE):** A backward elimination method selected an optimal subset of 15 features based on a model's performance, achieving an accuracy of 0.713.
*   **Forward Selection:** A forward-building method using F1-scores was implemented to iteratively add the best-performing features. Analysis showed diminishing returns after the first 10 features, with a final F1-score of 0.435 and accuracy of 0.738.

### Final selection
Based on a combined assessment of the statistical results and domain knowledge, a final set of 12 features was selected to build the predictive model. These include:
*   **Health:** `GENHLTH`, `BMI_recoded`, `CVDCRHD4`, `CHCKIDNY`, `DRVISITS_recoded`, `EXERANY2`, `MENTHLTH_recode`
*   **Demographic:** `AGE_recoded`
*   **Socioeconomic:** `INCOME_recoded`, `_EDUCAG`, `EMPLOY_recoded`
*   **Other:** `DECIDE`
```
  


