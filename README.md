# loan-model
This repository demonstrates the loan prediction through Python Pandas (Jupyter Notebook) and PySpark (Python File). It helped proactively identify the kind of loan whether it is good or bad.

## Description
This project is based on machine learning modeling in the finance and banking industries. It predicts whether the loan is good or bad on the banking dataset.

This banking dataset consists of the following subsets stored in the `data` directory:
1. Account
2. Card
3. Client
4. Disposition
5. District
6. Loan
7. Order
8. Transaction

This project consists the modeling which is based on two different platforms:

1. Jupyter Notebook: Data is stored locally and Jupyter Notebook runs in the local environment.
2. PySpark: Data is stored in Hive (Hadoop) and python code for cleaning, preparing, and modeling runs in the Spark environment.

### Feature engineering & Data Pre-processing-
1. Selected subsets of banking dataset for loan prediction exploration are Account, Disposition, Loan, Client, Card, Transaction.
2. Applied on date part to convert from number into a date.
3. Fetch correct dates for birth (Males and Females): Numerical Calculations/ String Operations.
4. Renaming of columns.
5. Performing join operation to combine the subsets into one dataset and extract feature and label out of the dataset.
6. For building the strong prediction of loan, including the columns like min, max, and mean for the cumulative M months before the loan date. Therefore,
    min1, max1, and mean1 are for the month before the loan; 
    min2, max2, and mean2 cover both of the two months before the loan (not just the second month before the loan); 
    min3, max3, and mean3 cover the three months before the loan.
7. Target label will be in 0 or 1 form as per Binary Classification, where '1' represents 'good loan' and '0' represents 'bad loan'.

### Modeling and Accuracy-
Random Forest Classifier is used for modeling the loan prediction. Train data is set to 75%, test data is set to 25% and the random state is set to 42 for retrieving the same samples each time when split the dataset into this proportion.

Receiver Operating Characteristic Area Under the Curve (ROC AUC) is used for evaluating and determining the performance metrics of the model. Jupyter Notebook consists of a detailed model.

### Working Details on PySpark-
The 2.3 Spark Version was used for running this code. Since `handleInvalid of vectorAssembler`, which is used for handling Null/NaN values, was not supportive in Spark 2.3, therefore used here `RFormula` to bypass this usage.

`requirements.txt` file consists of `pyspark` package name that is required to install to run this code.

## Further Implementation

This model was implemented by using RAD-ML Framework (RAD-ML is a proven methodology for developing sellable, reusable, and scalable machine learning assets). 

This model was further implemented by using IBM Cloud Pak for Data and Palantir recent product where feature engineering, data cleaning, data preprocessing, and modeling is done in IBM Cloud Pak for Data and AI-infused application was built on Palantir product to support operations for a credit analyst to predict loan on daily basis.

Implementations are not kept in this open-source GitHub to maintain confidentiality.

## Demo Screenshots

1. Jupyter Notebook: Detailed Implementation with steps and outputs are written in Notebook *loan_model.ipynb*
2. PySpark Code: Code is implemented in *spark_hive_model.py* and variables with hive store configuration settings are provided in the code.

Accuracy from this Model:
![PySpark Model Screenshot](https://github.com/Anshita1Saxena/loan-model/blob/main/demo-image/PySpark%20Model%20Screenshot.JPG)
