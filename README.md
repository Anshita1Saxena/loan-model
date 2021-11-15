# loan-model
This repository demonstrates the loan prediction through Python Pandas (Jupyter Notebook) and PySpark (Python File). It helped proactively identify the kind of loan whether it is good or bad.

## Description
This project is based on machine learning modelling on finance and banking industries. It predicts that the loan is good or bad on banking dataset.

This banking dataset consists of following subsets stored in `data` directory:
1. Account
2. Card
3. Client
4. Disposition
5. District
6. Loan
7. Order
8. Transaction

This project consists the modelling which is based on two different platforms:

1. Jupyter Notebook: Data is stored in local and Jupyter Notebook runs in local environment.
2. PySpark: Data is stored in Hive (Hadoop) and python code for cleaning, preparing and modelling runs in Spark environment.

### Feature engineering & Data Pre-processing-
1. Selected subsets of banking dataset for loan prediction exploration are Account, Disposition, Loan, Client, Card, Transaction.
2. Applied on date part to convert from number into date.
3. Fetch correct dates for birth (Males and Females): Numerical Calculations/ String Operations.
4. Renaming of columns.
5. Performing join operation to combine the subsets into one dataset and extract feature and label out of the dataset.
6. For building the strong prediction of loan, including the columns like min, max and mean for the cumulative M months before the loan date. Therefore,
    min1, max1, and mean1 are for the month before the loan; 
    min2, max2, and mean2 cover both of the two months before the loan (not just the second month before the loan); 
    min3, max3, and mean3 cover the three months before the loan.
7. Target label will be in 0 or 1 form as per Binary Classification, where '1' represents 'good loan' and '0' represents 'bad loan'.

### Modeling and Accuracy-
Random Forest Classifier is used for modeling the loan prediction. Train data is set to 75%, test data is set to 25% and the random state is set to 42 for retrieving same samples each time when split the dataset into this proportion.

Receiver Operating Characteristic Area Under the Curve (ROC AUC) is used for evaulation and determining performance metrics of model. Jupyter Notebook consists the detailed model.

### Working Details on PySpark-
The 2.3 Spark Version was used for running this code. Since handleInvalid, which is used for handling Null/NaN values, was not supportive in Spark 2.3, therefore used here RFormula to bypass this usage.
