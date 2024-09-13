# Machine Learning Models and Data Preprocessing

This repository contains a collection of Python notebooks and scripts for building machine learning models and preprocessing UKBB data. The following four notebooks are recommended for use by the wider audience:

1. Constructing a Regression Tree for data exploration
2. Building a Multilinear Regression Model
3. Preprocessing Dietary Data of UK BioBank 
4. Advanced experimental notebook that performs the following steps:
   - Applies a stratified train-test split to ensure well-balanced training set if stratification is enabled.
   - Builds multilinear models for both a simple and extended set of predictors.
   - Constructs a regression tree for the residuals, which represent the differences between the ground truth outcome and the values predicted by the simple linear model.


## Table of Contents

- Regression Tree ```RegressionTrees_generic.ipyb```
- Multilinear Regression Model ```multilinear_model_generic.ipyb```
- Preprocessing Dietary Data of UK BioBank ```food_bradbury.ipyb```
- Advanced experimental notebook ```multilinear_model_RegressionTrees.ipyb```

## 1. Regression Tree

### File: `RegressionTrees_generic.ipynb`

This notebook focuses on building a regression tree model and includes the following steps:

- **Importing the necessary libraries**: The first section of the notebook imports all required libraries.
- **Setting parameters**: The second section allows the user to adjust parameters to fine-tuning the model.
- **CSV data import and subsetting**: Data is imported and subsetted by the predictors and outcome variables defined in the parameters section.
- **Model construction and evaluation**: A regression tree is built using the `DecisionTreeRegressor` from `scikit-learn`. The regression tree is evaluated using various performance metrics, such as R^1, R², and by comparing predictions to actual outcomes.
- **Model interpretation**: The decision-making process within the tree is visualized using tools like feature importance and tree diagrams.

### Usage

We recommend running the notebook within an IDE such as VSCode for ease of use, as the model-building process depends on several adjustable parameters. After launching the notebook, run each cell to execute the code.

You can easily configure the parameters in the second section of the notebook. The parameter names and functions are self-explanatory, with additional comments included for clarification where needed.

Here, for illustration purposes, we have used the open dataset `diabetes_012_health_indicators_BRFSS2015.csv`, which is available at [this link on archive.ics.edu](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). Note, that the page referred by the link redirects to the dataset collection on Kaggle.

Since regression trees can be sensitive to the training set selection, we suggest the following:

- **Run the tree-building algorithm multiple times**: Use different values for the `RANDOM_STATE` parameter or adjust other tree parameters (see the parameters section). This helps you identify which predictors remain consistently significant across different runs. This approach is similar to building a random forest, but with more control and the ability to visualize individual trees.
  
- **Use the notebook for exploratory data analysis**: If the model's predictive strength is weak at the individual level, the notebook can still provide valuable insights for understanding broader patterns in the data. Instead of focusing on individual predictions, evaluate the model's performance on specific population subsets. For example, you can assess how well the model predicts the mean and median of the running example's outcome variable `Diabetes_012` (0 - no diabetes, 1 - prediabetes, 2 - diabetes) for individuals with high blood pressure, BMI > 31.5, and high cholesterol. Additionally, you can divide this group into two subgroups: those with and without difficulties walking and/or walking upstairs, to further explore the model’s predictive capability across these categories.

This analysis can be performed using the `eval_node` function described below.


### Parameter Setup

This section outlines the key parameters used for data loading, preprocessing, and setting up the working environment for tree-building algorithm. These parameters can be adjusted to suit the needs of the data analysis or debugging.

- **n_samples**:  
  Default: `None`  
  Description: Defines the number of data points to load. If set to `None`, all data points will be loaded. For debugging purposes, you can set this to a smaller number to load only a subset of the data.

- **remove_na**:  
  Default: `True`  
  Description: If set to `True`, all data points with at least one missing value are removed. This option is useful when working with non-imputed datasets.

- **gender_column**:  
  Default: `"Sex"`  
  Description: The name of the column in the dataset where gender or sex information is provided. This is important for subsetting the dataset based on gender.

- **gender**:  
  Default: `None`  
  Description: Used for subsetting the dataset by gender. If set to `None`, all data points will be considered. If a specific gender is required, this can be set accordingly.

- **iqr_coefficient**:  
  Default: `None`  
  Description: If set to `None`, no standard removal of outliers is performed. If an interquartile range (IQR) coefficient is provided, outliers will be removed using the formula:  
  `Q1 -/+ iqr_coefficient * (Q3 - Q1)`, where `Q1` is the 1st quartile and `Q3` is the 3rd quartile.

- **outcome**:  
  Example: `"Diabetes_012"`  
  Description: Specifies the column name in the dataset that contains the outcome variable, which in this case is `"Diabetes_012"`.


- **working_dir**:  
  Example: `"{home_directory}/PRIME/example_data"`  
  Description: The directory where input files will be loaded and output files will be saved.

- **input_file**:  
  Example: `"diabetes_012_health_indicators_BRFSS2015.csv"`  
  Description: Path to the input dataset. The file is expected to contain health indicators related to diabetes.



- **outcome_dir**:  
  Generated from the outcome name  
  Description: The part of the output directory name, where results related to the outcome will be stored. This part is generated by replacing special characters in the outcome name.

- **output_dir**:  
  Generated based on the outcome and gender  
  Description: The output directory path. If a specific gender is selected, it is appended to the directory name. The directory is created if it does not exist.

- **predictors**:  
  A list of predictor variables, for example:  
  `"HighBP"`, `"HighChol"`, `"CholCheck"`, `"BMI"`, `"Smoker"`, `"Stroke"`, `"HeartDiseaseorAttack"`, `"MentHlth"`, `"PhysActivity"`, `"DiffWalk"`, `"Fruits"`, `"Veggies"`, `"HvyAlcoholConsump"`, `"AnyHealthcare"`, `"NoDocbcCost"`, `"Sex"`, `"Age"`, `"Education"`, `"Income"`  
  Description: These are the predictor variables used in the model to predict the outcome. In this example they represent health and demographic attributes.

  
- **RANDOM_STATE**:  
  Default: `17`  
  Description: Used in the train-test split and tree-building process to ensure reproducibility. It allows the model to have consistent results by setting a fixed random seed. Additionally, it helps check how stable the tree is upon different train and test splits.



### Tree Building Algorithm Specific Parameters

- **m_samples_split**:  
  Default: `500`  
  Description: The minimum number of samples required to split a node in the decision tree.

- **m_samples_leaf**:  
  Default: `250`  
  Description: The minimum number of samples required to be present in a leaf node.

- **m_depth**:  
  Default: `6`  
  Description: The maximum depth of the tree. Limits how deep the tree can grow.


### Auxiliary Function: `eval_node`

The `eval_node` function, defined at the end of the notebook, is used to evaluate the performance of the regression tree at a specific node. In simple words, this function helps to see how good the regression tree predicts statistics, like mean value for an oitcome variable, for a given subgroup of population (or more generally, for a a subset of the given data set).

This function calculates key statistical properties for a node based on a given logical path, including median, quartiles, mean, and standard deviation for the data that flows to the left and right of the node. Additionally, it computes the percentage of values within a specified interval for both sides of the split.

By calling this function on both the training and validation sets, you can compare the statistics of the corresponding subsets and assess how closely they match, providing insight into the stability of the tree across the two datasets.



#### Parameters:
- **X**:  
  A DataFrame containing the feature variables.

- **y**:  
  A Series or DataFrame containing the target variable (or outcome).

- **logical_path**:  
  A list of conditions (predicates) representing the decision path to the current node in the tree. Each condition is a tuple with the following structure:  
  `(feature_name, comparison_operator, value)`  
  where the comparison operator is either `"le"` (less than or equal) or `"gt"` (greater than).

- **out_**:  
  A string representing the column name in `y` for which statistics are calculated.

- **interval_left**:  
  A tuple `(lower_bound, upper_bound)` representing the interval within which to evaluate the percentage of values in the left subtree. If `None`, this evaluation is skipped.

- **interval_right**:  
  A tuple `(lower_bound, upper_bound)` representing the interval within which to evaluate the percentage of values in the right subtree. If `None`, this evaluation is skipped.

#### Functionality:
1. The function first concatenates `X` and `y` into two DataFrames, `df_left` and `df_right`.
2. It iterates through the `logical_path` to split the data according to the specified predicates (e.g., whether a feature is less than or equal to a value or greater than a value).
3. After splitting the data, it calculates the following statistics for both the left and right splits:
   - Median
   - First quartile (Q1)
   - Third quartile (Q3)
   - Mean
   - Standard deviation (std)
4. Optionally, it computes the percentage of values within specified intervals (if `interval_left` and `interval_right` are provided).
5. The function returns a dictionary with the calculated statistics for both left and right subtrees.

#### Returns:
- A dictionary with the following keys:
  - `"median"`: A list of medians for the left and right subtrees.
  - `"q1"`: A list of first quartiles for the left and right subtrees.
  - `"q3"`: A list of third quartiles for the left and right subtrees.
  - `"mean"`: A list of means for the left and right subtrees.
  - `"std"`: A list of standard deviations for the left and right subtrees.
  - `"percentage_within_interval"`: A list of percentages of values within the specified intervals for the left and right subtrees (if intervals are provided).


#### Example: Usage of the `eval_node` Function

The following code demonstrates how to use the `eval_node` function to evaluate statistics at a specific node in a decision tree. The node's path is defined by a sequence of conditions (or predicates) based on certain features from the dataset. Each condition specifies a feature, a comparison operator (`"gt"` for greater than), and a threshold value. Variables 'HighBP', 'HighChol', 'DiffWalk' are binary, with 0 staying for 'no', and '1' staying for 'yes'.

```python
Walk_path = [['HighBP', "gt", 0.5],  ["BMI", "gt", 31.5], ["HighChol", "gt", 0.5], ['DiffWalk', "gt", 0.5]]

tr_node_stat = eval_node(X_train, y_train, Walk_path, outcome, None, None)

print(tr_node_stat)

interval_left = [tr_node_stat["mean"][0] - tr_node_stat["std"][0], tr_node_stat["mean"][0] + tr_node_stat["std"][0]]

interval_right = [tr_node_stat["mean"][1] - tr_node_stat["std"][1], tr_node_stat["mean"][1] + tr_node_stat["std"][1]]

val_node_stat  = eval_node(X_val, y_val, Walk_path, outcome, interval_left, interval_right)

print(val_node_stat)
```
##### Code Explanation:

1. **Define the node's path (`Walk_path`)**:  
   A list of conditions that define the path to the node in the regression tree. Each condition specifies a feature, a comparison operator (`"gt"` for greater than in this case), and a threshold value. Here, the conditions are:
   - `HighBP > 0.5`
   - `BMI > 31.5`
   - `HighChol > 0.5`
   - `DiffWalk > 0.5`

   This path will be used to filter the data and calculate statistics for the node.

2. **Evaluate training data node statistics (`tr_node_stat`)**:  
   The function `eval_node` is called with the training data (`X_train` and `y_train`), the defined path (`Walk_path`), and the outcome variable (`outcome`). The result is stored in `tr_node_stat`, which contains statistics for the data that flows left and right of the node.

   Since no interval is provided for left or right splits (both are `None`), only basic statistics like median, quartiles, mean, and standard deviation are calculated.

    Here is an example of the tr_node_stat dictionary content:

   ```'median': [0.0, 2.0], 'q1': [0.0, 0.0], 'q3': [2.0, 2.0], 'mean': [0.7661164746184989, 1.1015930267508265], 'std': [0.9529014758766701, 0.975833275092657], 'percentage_within_interval': [None, None]```

   The first values in the lists represent the corresponding statistics for the group of individuals with high blood pressure, BMI > 31.5, high cholesterol, and **not  having difficulties walking**.

The second values in the lists represent the statistics for the group of individuals with high blood pressure, BMI > 31.5, high cholesterol, and **having difficulties walking**.


3. **Calculate intervals for validation evaluation**:
    Two intervals (interval_left_waist and interval_right_waist) are calculated based on the mean and standard deviation from the left and right splits of the training data. These intervals are defined as:

    For the left split: ```[mean_left - std_left, mean_left + std_left]```
    For the right split: ```[mean_right - std_right, mean_right + std_right]``` 
    These intervals represent a range within one standard deviation of the mean and will be used for validation.

4. **Evaluate node statistics on the validation set (val_node_stat)**:
    The eval_node function is called again, this time with the validation data (X_val and y_val), the same logical path (Walk_path), and the previously calculated intervals (interval_left_waist, interval_right_waist). This evaluates how well the validation data fits within these intervals. The results are printed.

    Here is an example of the ```val_node_stat``` dictionary:

    ```'median': [0.0, 2.0], 'q1': [0.0, 0.0], 'q3': [2.0, 2.0], 'mean': [0.7474402730375427, 1.0825932504440496], 'std': [0.9484569424981746, 0.9797228176304701], 'percentage_within_interval': [64.47409246044059, 47.55772646536412]```

    We can observe that for both subcategories of the population—those with and without difficulties walking—within the group of individuals with high blood pressure, BMI > 31.5, and high cholesterol, the mean and median values of the training set are quite close to those of the validation set.
    
    Mean not having difficulties, training: 0.766; validation: 0.747

    Mean having difficulties, training: 1.10; validation:  1.08

    Recall, that 0 denotes no diabetes or prediabetes, 1 is for prediabetes and 2 is for diabetes.
    
    This indicates that the prediction of these statistics is quite accurate for this group.


  



## 2. Multilinear Regression Model

### File: multilinear_model_generic.ipyb

This Jupyter Notebook demonstrates how to build and evaluate a multilinear regression model using Python. We assume that data preprocessing, and if necessary, adding transformations such as logarithms, squares, or other nonlinear terms, has been performed prior to model building during the data preprocessing phase. Initially, we included the creation of nonlinear terms within the model-building script, but we later moved it to the preprocessing phase for improved efficiency.

As the tree-building notebook, this notebook covers the following steps:

- **Importing the necessary libraries**
- **Setting parameters**
- **CSV data import and subsetting**
- **Model construction and evaluation**: Multilinear models are built using `LinearModel` from `scikit-learn` and `OLS` from `statsmodels`. We have implemented two main functions for model building: `model_builder` and `lasso_model_builder`. In `lasso_model_builder`, a Lasso cross-validation (`lasso-cv`) from `scikit-learn` is used to identify the most significant features and eliminate collinearities through cross-validation before building the Ordinary Least Square (OLS) model in `statsmodels`. The OLS model is then constructed using only the selected features.

By running `model_builder` with simple predictors and `lasso_model_builder` with all predictors, we can assess how much the additional predictors improve the model’s performance compared to using only the simple predictors.


  

### Parameter Setup

This section describes the key parameters that can be adjusted to control the behavior of the model-building pipeline. The parameters follow conventions similar to those used in the Regression Tree Builder. Each parameter is explained in detail below:


- **n_samples**:  
  Default: `None`  
  Description: If `None`, all data points are used. If set to an integer, only the first `n_samples` data points are loaded, typically used for debugging purposes.

- **lasso_epsilon**:  
  Default: `0.01`  
  Description: Controls the step for Lasso cross-validation to remove collinearities. If `None`, the Lasso step is bypassed.

- **p_val**:  
  Default: `0.05`  
  Description: The significance level for statistical tests.

- **age_limit**:  
  Default: `0`  
  Description: A threshold for filtering data by age. Data points with age below this value will be excluded.

- **gender**:  
  Default: `None`  
  Description: Used for filtering data by gender. If `None`, all genders are included.

- **gender_column**:  
  Default: `"Sex"`  
  Description: The name of the column that contains gender or sex information in the dataset.

- **iqr_coefficient**:  
  Default: `1.5`  
  Description: Used for removing outliers based on the interquartile range (IQR). If `None`, no outlier removal is performed. Outliers are detected and removed using the formula:  
  `Q1 -/+ iqr_coefficient * (Q3 - Q1)`.

- **simple_predictors**:  
  Example: `["Age", "BMI"]`  
  Description: A short list of predictor variables used for basic modeling.

- **predictors**:  
  Example:  
  `["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "MentHlth", "PhysActivity", "DiffWalk", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "Sex", "Age", "Education", "Income"]`  
  Description: The full list of predictor variables used for model building. This list must include the list of simple perdictors above.

- **binarys**:  
  Example: `["HighBP", "HighChol"]`  
  Description: A list of binary predictors, where each variable can take only two possible values (e.g., 0 or 1). This list helps identify cases where all values for a given predictor are either 'zero' (e.g., all individuals in a selected population have low cholesterol) or 'one'. In such cases, the corresponding column will be removed from the dataset before building the model. You can include all binary predictors or only a subset that the researcher considers important to check.


- **outcome**:  
  Example: `"Diabetes_012"`  
  Description: The outcome variable (target) to be predicted by the model.

- **HSCD**:  
  Default: `'HC1'`  
  Description: If `None`, the model is heteroscedasticity-unaware. Otherwise, it enables heteroscedasticity-aware modeling.

- **remove_na**:  
  Default: `True`  
  Description: If `True`, data points with missing values are removed. This is required for datasets with missing values.

- **working_dir**:  
  Default: `"{home_directory}/PRIME/example_data"`  
  Description: The working directory where input and output files are located. It contains the input CSV file and will store the output files (such as models).

- **input_file**:  
  Example: `"diabetes_012_health_indicators_BRFSS2015.csv"`  
  Description: The path to the input CSV file, which in the example contains the diabetes health indicators dataset. This dataset can be downloaded from the following sources:
  - [UCI Dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
  - [Kaggle Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download)



### Usage
We recommend running the notebook within an IDE such as VSCode. Open the notebook, set up the parameters and execute the cells to build and evaluate the model.

### Key functions 1:  `lasso_model_builder`

The `lasso_model_builder` function creates a predictive model using Lasso regression, followed by an Ordinary Least Squares (OLS) model built from selected features. The function saves intermediate and final results in the specified directory, allowing users to evaluate model performance and feature importance.

#### Parameters:
- **df_source**:  
  A DataFrame containing the input data, including both the predictors and the outcome variable.

- **predictors**:  
  A list of feature names (columns) to be used as predictors in the model.

- **outcome_var**:  
  The name of the column in `df_source` representing the outcome variable (target).

- **result_dir**:  
  A string representing the directory where intermediate and final results will be saved. If the directory does not exist, it will be created.

- **hcd**:  
  Specifies whether to account for heteroscedasticity when building the OLS model. If `None`, heteroscedasticity is ignored.

#### Function Workflow:

1. **Directory Setup**:  
   If the specified `result_dir` does not exist, it is created.

2. **Data Preparation**:  
   - The outcome variable `y` is extracted from `df_source`.
   - The predictor variables `X` are selected from the specified `predictors`.

3. **Lasso Model**:  
   - A Lasso regression model with 100-fold cross-validation is applied to the data.
   - The regularization strength (`alpha`), intercept, and coefficients of the model are saved to a text file.  
   - The R² score of the Lasso model is also calculated and included in the summary file.

4. **Feature Selection**:  
   - The function identifies predictors with non-zero coefficients (determined by a threshold `lasso_epsilon`). These predictors are considered significant and are selected for the next step.

5. **Statistics**:  
   - For each predictor, the mean, standard deviation, minimum, and maximum values are computed and stored in a DataFrame.

6. **OLS Model**:  
   - An OLS model is built using all the predictors, including an intercept. If heteroscedasticity correction (`hcd`) is provided, the model adjusts for it.
   - The OLS coefficients and p-values are saved in a CSV file along with the basic statistics for each predictor.

7. **Final OLS Model on Selected Predictors**:  
   - The second OLS model is built using only the significant predictors identified by the Lasso model.
   - The results of this final OLS model, including the coefficients and p-values, are saved in a CSV file.

8. **Outputs**:  
   - The function returns a DataFrame of selected features, including their OLS coefficients, p-values, and statistics.
   - The final OLS model (`model_2`) is also returned.

#### Files Created:
- `intermediate_results_lasso_technical_summary.txt`: Contains the summary of the Lasso model, including the best `alpha`, coefficients, and R² score.
- `intermediate_results_lasso_readable.csv`: Stores the OLS coefficients, p-values, and basic statistics for all predictors.
- `final_aftre_lasso_OSL_model_summary.txt`: A summary of the OLS model built on the selected predictors.
- `final_after_lasso_OSL_model_with_stats.csv`: The final selected predictors with OLS coefficients and related statistics.

#### Example Usage:
```model_table, model = lasso_model_builder(df, global_predictors, outcome, f"{output_dir}",  HSCD)```

### Key functions 2: `model_builder`

The `model_builder` function creates an Ordinary Least Squares (OLS) regression model from a set of predictors and an outcome variable. The function also calculates and stores basic statistics (mean, standard deviation, minimum, and maximum) for each predictor, alongside the OLS model coefficients and p-values. Results are saved to the specified directory.

#### Parameters:
- **df_source**:  
  A DataFrame containing the dataset, including both the predictor variables and the outcome variable.

- **predictors**:  
  A list of column names to be used as predictors (features) in the model.

- **outcome_var**:  
  The name of the column in `df_source` representing the outcome variable (target).

- **result_dir**:  
  A string specifying the directory where the model summary and results will be saved. If the directory does not exist, it will be created.

- **hcd**:  
  Used for heteroscedasticity correction. If set to `None`, the model is heteroscedasticity-unaware. Otherwise, a specific covariance type can be provided (e.g., `HC1`, `HC3`) for heteroscedasticity-consistent standard errors.

#### Function Workflow:

1. **Directory Setup**:  
   If the specified `result_dir` does not exist, it is created.

2. **Data Preparation**:  
   - The outcome variable `y` is extracted from the DataFrame.
   - The predictor variables `X` are selected from the specified `predictors`.
   - The function calculates the mean, standard deviation, minimum, and maximum for each predictor. These statistics are stored in a DataFrame.

3. **Data Standardization**:  
   Each predictor is standardized by subtracting its mean and dividing by its standard deviation.

4. **OLS Model**:  
   - An OLS model is fitted to the standardized predictors (with an intercept) and the outcome variable. If heteroscedasticity correction (`hcd`) is provided, the model adjusts for it.
   - The summary of the OLS model, including coefficients, R², and other model diagnostics, is written to a file in the result directory.

5. **Results Compilation**:  
   - The coefficients and p-values from the OLS model are saved in a DataFrame along with the previously calculated statistics (mean, standard deviation, etc.) for each predictor.
   - This DataFrame is saved as a CSV file in the result directory.

6. **Outputs**:  
   The function returns:
   - `df_results`: A DataFrame containing OLS coefficients, p-values, and statistics for each predictor.
   - `model_1`: The fitted OLS model.

#### Files Created:
- `simple_ols_model_summary.txt`: Contains the summary of the OLS model, including coefficients, R², and model diagnostics.
- `simple_ols_model_with_stats.csv`: Contains the OLS model results (coefficients and p-values) along with the mean, standard deviation, minimum, and maximum values for each predictor.

#### Example Usage:
```model_table_simple, model_simple = model_builder(df, simple_predictors, outcome, f"{output_dir}",  HSCD)``


## 3. Preprocessing Dietary Data

### File: ```food_bradbury.ipynb```

This notebook is designed for researchers working with dietary data from the UK Biobank (UKBB). The dietary data collection in UKBB was not specifically tailored for detailed dietary research, with participants answering questionnaires that were sometimes challenging to interpret precisely. For example, responses to questions like “how many heaped tablespoons of cooked vegetables do you usually eat per day” are difficult to reliably convert into a consistent metric for vegetable consumption.

To address this challenge, we developed a Python script that translates UKBB dietary and alcohol intake responses into grams per day, using conversion tables proposed by Bradbury et al. in their studies:

- Bradbury, K. E., Murphy, N., & Key, T. J. (2020, February). Diet and colorectal cancer in UK Biobank: a prospective study. *International Journal of Epidemiology, 49*(1). https://doi.org/10.1093/ije/dyz064
- Bradbury, K. E., Young, H. J., Guo, W., & Key, T. J. (2018, February 01). Dietary assessment in UK Biobank: an evaluation of the performance of the touchscreen dietary questionnaire. *Journal of Nutritional Science, 7*. https://doi.org/10.1017/jns.2017.66
  


### Usage
You can run this Python script in an IDE like VSCode. Ensure you pass the input file path as an argument when running the script, for example:
`python preprocess_dietary_data.py --input data/dietary_data.csv`

  
## 4. Advanced Experimental Notebook
### File: ```mutilinear_model_RegressionTrees.ipynb```

This notebook performs the following steps:
   - if stratification is enabled, applies a stratified train-test split to ensure balanced subsets across categories.
   - Builds multilinear models for both a simple and extended set of predictors.
   - Constructs a regression tree for the residuals, which represent the differences between the ground truth outcome and the values predicted by the simple linear model.

### Setup parameters.

In addition to the parameters listed in the `RegressionTrees_generic` notebook, this notebook includes several parameters for configuring the stratified train-test procedure. These parameters can be adjusted to explore the stability of the model.

#### Parameters for Train-Test Split:
- **stratify**:
  Default: `True`  
  `train_test_split` function is used.

- **percentage_test**:  
  Default: `0.25`  
  Description: The percentage of data points to be included in the validation set.

- **n_splits**:  
  Default: `4`  
  Description: The number of splits to create for the stratified train-test procedure.

- **n_fold**:  
  Default: `1`  
  Description: The index of the fold to use when building the model in the stratified train-test split.

- **quantiles**:  
  Default: `[0.2, 0.4, 0.6, 0.8, 1]`  
  Description: The quantiles used for stratifying the train-test split for non-categorical data. These quantiles help ensure balanced data subsets.

#### Categorical Predictors for Stratification:
- **cat_predictors**:  
  A list of categorical predictors that assist with stratifying the training set. In the provided dataset, all predictors except BMI, Mental Health (count depressed or stressed days last month) and age are categorical. While the example dataset documentation suggests age is categorized, since it exceeds 10 categories, we recommend treating age as a non-categorical variable.  

  Example list of categprical predictors:
  ```cat_predictors = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
              "HeartDiseaseorAttack", "MentHlth", "PhysActivity", "DiffWalk", "Fruits", "Veggies", "HvyAlcoholConsump", 
             "AnyHealthcare", "NoDocbcCost", "Sex",  "Education", "Income"]```

  
### Function ```my_stratification```

This function configures the selection of training and validation sets using a stratified train-test split. It ensures balanced subsets across categories, which is crucial for model stability. The function allows you to adjust parameters such as the percentage of data in the validation set, the number of splits, and the fold index to be used. 

#### Parameters:
- **percentage_test** (`float`):  
  The percentage of data points to include in the validation set.

- **n_splits** (`int`):  
  The number of splits to create for the stratified train-test procedure.

- **n_fold** (`int`):  
  The index of the fold to be used when performing the stratified train-test split. This allows you to experiment with different folds.

- **quantiles** (`list of float`):  
  For non-categorical data, this parameter defines the quantiles to use for the stratified train-test split. It ensures that the data is split proportionally across the quantiles.

- **cat_predictors** (`list of str`):  
  A list of categorical predictors that guide stratification. These predictors help ensure the distribution of categorical values is maintained in both the training and validation sets.


### Main steps

The key steps of the notebook are organized into separate cells, each with comments explaining the purpose of the step:

- **Building and evaluating a multilinear model** using simple predictors.
- **Building and evaluating a multilinear model** for the residuals, which represent the difference between the ground truth outcome and the predictions from the simple model.
- **Building and evaluating an alternative regression tree model** for the residuals to capture any nonlinear dependencies.

The results are displayed "on the fly" after each corresponding cell in the notebook. 

Additionally, the notebook includes function definitions for visualizing the regression tree model and identifying its most significant features.


# Requirements

To install the necessary dependencies, you can use the following command:
`pip install -r requirements.txt` where the requirements.txt should contain the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `statsmodels`
- `os`
- `re`
- `shutil`
- `copy`


BSD License

Copyright (c) 2024 Machine2Learn BV.
All rights reserved.
