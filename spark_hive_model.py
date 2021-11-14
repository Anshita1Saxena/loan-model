"""
The model is designed for checking the loan eligibility of customer.
"""
__author__ = "Anshita Saxena"
__copyright__ = "(c) Copyright IBM 2020"
__contributors__ = "Ryan P Chiang, Ramesh Sigamani, Debabrata Ghosh"
__credits__ = ["IBM CAD-DICE Team"]
__email__ = "anshita333saxena@gmail.com"
__status__ = "Development"

# Importing the necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

# Initialize spark configuration properties
memory = '30g'
master = 'yarn'
pyspark_submit_args = ' --master ' + master + ' --driver-memory ' + memory + \
                      ' pyspark-shell'
hadoop_conf_dir = '/etc/hadoop/conf'
yarn_conf_dir = '/etc/hadoop/conf'
spark_home = '/usr/hdp/current/spark2-client/'

# Setting environment variables
os.environ["SPARK_HOME"] = spark_home
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
os.environ["HADOOP_CONF_DIR"] = hadoop_conf_dir
os.environ["YARN_CONF_DIR"] = yarn_conf_dir
print(os.environ)

# Creating spark hive session connectivity
spark = SparkSession.builder.config("hive.metastore.uris",
                                    "thrift://master.hdp.cad.aws.internal:9083") \
    .enableHiveSupport().getOrCreate()
print(spark.sparkContext._conf.getAll())

# Loading data from hive
account_df = spark.sql(
    "SELECT ac_id as account_id, dstc_id as district_id, frq as frequency, "
    "ac_opn_dt as date FROM banking_dataset_test.account")
account_df.show(n=2)

card_df = spark.sql(
    "SELECT card_id, card_dspl_id as disp_id, card_tp as type, "
    "card_issu_dt as issued FROM banking_dataset_test.card_cz")
card_df.show(n=2)

client_df = spark.sql(
    "SELECT clnt_identn_nbr as client_id, dob as birth_number, "
    "dstc_id as district_id FROM banking_dataset_test.client_cz")
client_df.show(n=2)

disposition_df = spark.sql(
    "SELECT dspn_id as disp_id, clnt_identn_nbr as client_id, "
    "ac_id as account_id, dspn_tp as type FROM banking_dataset_test.disp_cz")
disposition_df.show(n=2)

loan_df = spark.sql(
    "SELECT loan_id, ac_id as account_id, loan_dt as date, "
    "loan_amt as amount, loan_drtn as duration, mo_pymt as payments, "
    "loan_st as status FROM banking_dataset_test.loan_cz")
loan_df.show(n=2)

order_df = spark.sql(
    "SELECT ordr_id as order_id, ac_id as account_id, "
    "bnk_code_receipient as bank_to, ac_nbr_receipient as account_to, "
    "amt as amount, pymt_characterization as k_symbol "
    "FROM banking_dataset_test.payment_order_cz")
order_df.show(n=2)

transaction_df = spark.sql(
    "SELECT txn_id as trans_id, ac_id as account_id, txn_dt as date, "
    "txn_tp as type, mode_of_txn as operation, txn_amt as amount, "
    "txn_bal as balance, tranaction_characterization as k_symbol, "
    "partner_bnk as bank, partner_bnk_ac_nbr as account "
    "FROM banking_dataset_test.trans")
transaction_df.show(n=2)

# Checking the categories of Status
print("\nLoan Status Categories:")
loan_df.groupBy('status').count().orderBy('count', ascending=False).show()

# Rephrasing text in english with correct time-date format for Account
# Formatting Date
print("Account Date Conversion:")
account_df = account_df.withColumn("date", concat(lit("19"), col("date")))
account_df = account_df.select('account_id', 'district_id', 'frequency',
                               'date', from_unixtime(
        unix_timestamp('date', 'yyyyMMdd')).cast("timestamp").alias('dt'))
account_df = account_df.drop('date')
account_df = account_df.selectExpr('account_id', 'district_id', 'frequency',
                                   'dt as date')
account_df = account_df.withColumn('frequency',
                                   regexp_replace('frequency', '"', ''))
account_df.show(n=2)
# Rephrasing text
print("\nAccount Text Conversion:")
account_df = account_df.withColumn('frequency', regexp_replace('frequency',
                                                               'POPLATEK MESICNE',
                                                               'monthly_issuance'))
account_df = account_df.withColumn('frequency', regexp_replace('frequency',
                                                               'POPLATEK TYDNE',
                                                               'weekly_issuance'))
account_df = account_df.withColumn('frequency', regexp_replace('frequency',
                                                               'POPLATEK PO OBRATU',
                                                               'issuance_after_transaction'))
account_df.show(n=8)

# Formatting date in Card
print("Card Date Conversion:")
card_df = card_df.withColumn("issued", concat(lit("19"), col("issued")))
card_df = card_df.select('card_id', 'disp_id', 'type', 'issued', from_unixtime(
    unix_timestamp('issued', 'yyyyMMdd')).cast('timestamp').alias(
    'issued_date'))
card_df = card_df.drop('issued')
card_df = card_df.selectExpr('card_id', 'disp_id', 'type',
                             'issued_date as issued')
# Removing double quotes from 'type' column
card_df = card_df.withColumn('type', regexp_replace('type', '"', ''))
print("Card Dataframe:")
card_df.show(n=2)

# Modifying date format for Client
print("Client Dataframe:")
# Removing double quotes from column 'birth_number'
client_df = client_df.withColumn('birth_number', regexp_replace('birth_number',
                                                                '"', ''))
# Applying logic to extract birth month of client
client_df = client_df.withColumn('birth_num', expr(
    'substring(birth_number, 3, length(birth_number)-4)'))
client_df.show(n=2)
# Assigning gender to Male & Female acc. to greater than 50 logic
client_df = client_df.withColumn('gender',
                                 when(client_df.birth_num > 50, 'F').otherwise(
                                     'M'))
client_df.show(n=2)
# Modifying birth month of Female from 50-62 to 1-12 respectively
client_df = client_df.withColumn('birth_month', when(client_df.gender == 'F', (
        client_df.birth_num - 50).cast('int')).otherwise(
    client_df.birth_num))
client_df.show(n=2)
# Applying logic to format single-digit month to double-digit
client_df = client_df.withColumn('birth_month_length', length('birth_month'))
client_df = client_df.withColumn('birth_mon',
                                 when(client_df.birth_month_length == 1,
                                      concat(lit('0'),
                                             client_df.birth_month)).otherwise(
                                     client_df.birth_month))
client_df.show(n=2)
# Appending 19 to the year for preparing meaningful data
client_df = client_df.withColumn('birth_number',
                                 concat(lit('19'), col('birth_number')))
client_df.show(n=2)
# Applying logic to select substring and format date meaningful
client_df = client_df.withColumn('birth_new', concat(
    expr('substring(birth_number, 1, length(birth_number)-4)'),
    col('birth_mon'),
    expr('substring(birth_number, 7, length(birth_number))')))
client_df.show(n=2)
# Select required columns for the Client
client_df = client_df.select('client_id', 'birth_new', 'district_id', 'gender',
                             from_unixtime(
                                 unix_timestamp('birth_new', 'yyyyMMdd')).cast(
                                 'timestamp').alias('birth_number'))
client_df.show(n=2)
# Dropping unnecessary column from Client
client_df = client_df.drop('birth_new')
client_df.show(n=2)
# Final columns of Client
client_df = client_df.selectExpr('client_id', 'birth_number', 'district_id',
                                 'gender')
client_df.show(n=8)

# Rephrasing Disposition into english phrases with removal of double quotes
print("Disposition Dataframe:")
# Removing the double quotes from Disposition
disposition_df = disposition_df.withColumn('type',
                                           regexp_replace('type', '"', ''))
# Rephrasing into english phrases
disposition_df = disposition_df.withColumn('type',
                                           regexp_replace('type', 'OWNER',
                                                          'owner'))
disposition_df = disposition_df.withColumn('type',
                                           regexp_replace('type', 'DISPONENT',
                                                          'disponent'))
disposition_df.show(n=2)

# Removing the double quotes and format date from Loan
print("Loan Dataframe:")
# Removing the double quotes
loan_df = loan_df.withColumn('status', regexp_replace('status', '"', ''))
# Formatting the date
loan_df = loan_df.withColumn("date", concat(lit("19"), col("date")))
loan_df = loan_df.select('loan_id', 'account_id', 'date', 'amount', 'duration',
                         'payments', 'status', from_unixtime(
        unix_timestamp('date', 'yyyyMMdd')).cast('timestamp').alias(
        'loan_date'))
# Dropping the unnecessary column
loan_df = loan_df.drop('date')
loan_df = loan_df.selectExpr('loan_id', 'account_id', 'loan_date as date',
                             'amount', 'duration', 'payments', 'status')
loan_df.show(n=2)

# Removing the double quotes from Order and rephrasing into english phrases
print("Order Dataframe:")
# Removing the double quotes
order_df = order_df.withColumn('bank_to', regexp_replace('bank_to', '"', ''))
order_df = order_df.withColumn('account_to',
                               regexp_replace('account_to', '"', ''))
order_df = order_df.withColumn('k_symbol', regexp_replace('k_symbol', '"', ''))
# Rephrasing into english phrases
order_df = order_df.withColumn('k_symbol',
                               regexp_replace('k_symbol', 'POJISTNE',
                                              'insurrance_paymt'))
order_df = order_df.withColumn('k_symbol', regexp_replace('k_symbol', 'SIPO',
                                                          'household_paymt'))
order_df = order_df.withColumn('k_symbol',
                               regexp_replace('k_symbol', 'LEASING',
                                              'leasing_paymt'))
order_df = order_df.withColumn('k_symbol', regexp_replace('k_symbol', 'UVER',
                                                          'loan_paymt'))
order_df = order_df.selectExpr('order_id', 'account_id', 'bank_to',
                               'account_to', 'amount',
                               'k_symbol as category_of_payment')
order_df.show(n=2)

# Remove the double quotes, format the dates and rephrase into english phrases
print("Transaction Dataframe:")
# Remove the double quotes
transaction_df = transaction_df.withColumn('type',
                                           regexp_replace('type', '"', ''))
transaction_df = transaction_df.withColumn('operation',
                                           regexp_replace('operation', '"',
                                                          ''))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol', '"', ''))
# Format the dates
transaction_df = transaction_df.withColumn("date",
                                           concat(lit("19"), col("date")))
transaction_df = transaction_df.select('trans_id', 'account_id', 'date',
                                       'type', 'operation', 'amount',
                                       'balance', 'k_symbol', 'bank',
                                       'account', from_unixtime(
        unix_timestamp('date', 'yyyyMMdd')).cast('timestamp').alias(
        'transaction_date'))
transaction_df = transaction_df.drop('date')
transaction_df = transaction_df.selectExpr('trans_id', 'account_id',
                                           'transaction_date as date', 'type',
                                           'operation', 'amount', 'balance',
                                           'k_symbol', 'bank', 'account')
# Rephrase into english phrases
transaction_df = transaction_df.withColumn('type',
                                           regexp_replace('type', 'PRIJEM',
                                                          'credit'))
transaction_df = transaction_df.withColumn('type',
                                           regexp_replace('type', 'VYDAJ',
                                                          'withdrawal'))
transaction_df = transaction_df.withColumn('operation',
                                           regexp_replace('operation',
                                                          'VYBER KARTOU',
                                                          'creditcard_wd'))
transaction_df = transaction_df.withColumn('operation',
                                           regexp_replace('operation', 'VKLAD',
                                                          'credit_in_cash'))
transaction_df = transaction_df.withColumn('operation',
                                           regexp_replace('operation',
                                                          'PREVOD Z UCTU',
                                                          'coll_from_bank'))
transaction_df = transaction_df.withColumn('operation',
                                           regexp_replace('operation', 'VYBER',
                                                          'cash_wd'))
transaction_df = transaction_df.withColumn('operation',
                                           regexp_replace('operation',
                                                          'PREVOD NA UCET',
                                                          'remi_to_bank'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol',
                                                          'POJISTNE',
                                                          'insurrance_paymt'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol', 'SLUZBY',
                                                          'paymt_for_stmt'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol', 'UROK',
                                                          'interest_credited'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol',
                                                          'SANKC. UROK',
                                                          'sanction_interest'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol', 'SIPO',
                                                          'household'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol', 'DUCHOD',
                                                          'pension'))
transaction_df = transaction_df.withColumn('k_symbol',
                                           regexp_replace('k_symbol', 'UVER',
                                                          'loan_paymt'))
# Selecting the necessary columns
transaction_df = transaction_df.selectExpr('trans_id', 'account_id', 'date',
                                           'type', 'operation', 'amount',
                                           'balance',
                                           'k_symbol as category_of_payment',
                                           'bank', 'account')
transaction_df.show(n=2)

# Data Processing Part Starts
# Merging the disposition with Account
features = account_df.join(disposition_df, on='account_id', how='outer')
# Merging features with Loan
features = features.selectExpr('account_id', 'district_id', 'frequency',
                               'date as date_acct', 'disp_id', 'client_id',
                               'type').join(
    loan_df.selectExpr('loan_id', 'account_id', 'date as date_loan', 'amount',
                       'duration', 'payments', 'status'), on='account_id',
    how='left')
# Merging features with Client
features = features.selectExpr('account_id', 'district_id as district_id_bank',
                               'frequency', 'date_acct', 'disp_id',
                               'client_id', 'type', 'loan_id', 'date_loan',
                               'amount', 'duration', 'payments',
                               'status').join(
    client_df.selectExpr('client_id', 'birth_number',
                         'district_id as district_id_client', 'gender'),
    on='client_id', how='outer')
# Merging features with Card
features = features.selectExpr('account_id', 'district_id_bank', 'frequency',
                               'date_acct', 'disp_id', 'client_id',
                               'type as type_disp', 'loan_id', 'date_loan',
                               'amount', 'duration', 'payments', 'status',
                               'birth_number', 'district_id_client',
                               'gender').join(
    card_df.selectExpr('card_id', 'disp_id', 'type as type_card',
                       'issued as date_card'), on='disp_id', how='outer')
print('Filtering clients without loan_id:')
features = features.filter(features.loan_id.isNotNull())

"""
Append columns of minbalM, maxbalM, meanbalM for the cumulative M months before loan start date.
(note:technically it's M*30days rather than M months due to varying days/month)
"""


def addcols_monbalstats(features, trans, M):
    # Merging features with transaction
    trans_acctdate = trans.join(features.selectExpr('account_id', 'date_loan'),
                                on='account_id', how='inner')
    # Calculate date difference between transaction date and loan date
    trans_acctdate = trans_acctdate.withColumn('datediff', datediff(
        to_date(trans_acctdate.date_loan), to_date(trans_acctdate.date)))
    # Drop the rows with negative date difference
    trans_acctdate = trans_acctdate.withColumn('lt_0', when(
        trans_acctdate.datediff < 0, 'Y').otherwise('N'))
    trans_acctdate = trans_acctdate.filter(trans_acctdate.lt_0 == 'N')
    trans_acctdate = trans_acctdate.drop('lt_0')
    trans_acctdate = trans_acctdate.withColumn('lt_30', when(
        trans_acctdate.datediff > M * 30, 'Y').otherwise('N'))
    trans_acctdate = trans_acctdate.filter(trans_acctdate.lt_30 == 'N')
    trans_acctdate = trans_acctdate.drop('lt_30')
    # Append column names with month number
    _min_ = 'min' + str(M)
    _max_ = 'max' + str(M)
    _mean_ = 'mean' + str(M)
    expr = [min(col('balance')).alias(_min_), max(col('balance')).alias(_max_),
            mean(col('balance')).alias(_mean_)]
    # Extract result with unique account id
    monbalstats = trans_acctdate.groupBy('account_id').agg(*expr)
    # Merging features with modified result
    features = features.join(monbalstats, on='account_id', how='left')
    return features


# Calculate Min, Max and Mean before six months of applying for loan
features = addcols_monbalstats(features, transaction_df, 1)
features = addcols_monbalstats(features, transaction_df, 2)
features = addcols_monbalstats(features, transaction_df, 3)
features = addcols_monbalstats(features, transaction_df, 4)
features = addcols_monbalstats(features, transaction_df, 5)
features = addcols_monbalstats(features, transaction_df, 6)
# Displaying the names of features
print('Column Names: ', features.columns)

# Categorise the good and bad loans and convert into numerical values
print("\nLoan Status Categories:")
loan_df.groupBy('status').count().orderBy('count', ascending=False).show()
features = features.withColumn('response', when(
    (features.status == 'A') | (features.status == 'C'), 1).otherwise(0))
features = features.withColumn('gen',
                               when(features.gender == 'F', 1).otherwise(0))
features = features.withColumn('has_card', when(features.date_card.isNotNull(),
                                                1).otherwise(0))
# Dropping the unnecessary columns
features = features.drop('status')
features = features.drop('gender')
features = features.drop('date_card')

# Format the date
df_features = features.select(
    unix_timestamp(col('date_acct'), format='yyyy-MM-dd HH:mm:ss').alias(
        'date_acct_'),
    unix_timestamp(col('date_loan'), format='yyyy-MM-dd HH:mm:ss').alias(
        'date_loan_'), 'amount', 'duration', 'payments',
    unix_timestamp(col('birth_number'), format='yyyy-MM-dd HH:mm:ss').alias(
        'birth_number_'), 'min1', 'max1', 'mean1', 'min2', 'max2', 'mean2',
    'min3', 'max3', 'mean3', 'min4', 'max4', 'mean4', 'min5', 'max5', 'mean5',
    'min6', 'max6', 'mean6', 'gen', 'has_card', 'frequency', 'type_disp',
    'response')
df_features.show(n=2)

# Displaying the data types for columns
print(df_features.dtypes)
# Casting the string columns into numerical columns
df_features = df_features.withColumn("amount",
                                     df_features["amount"].cast(IntegerType()))
df_features = df_features.withColumn("duration", df_features["duration"].cast(
    IntegerType()))
df_features = df_features.withColumn("payments",
                                     df_features["payments"].cast('float'))
df_features = df_features.withColumn("min1", df_features["min1"].cast('float'))
df_features = df_features.withColumn("max1", df_features["max1"].cast('float'))
df_features = df_features.withColumn("min2", df_features["min2"].cast('float'))
df_features = df_features.withColumn("max2", df_features["max2"].cast('float'))
df_features = df_features.withColumn("min3", df_features["min3"].cast('float'))
df_features = df_features.withColumn("max3", df_features["max3"].cast('float'))
df_features = df_features.withColumn("min4", df_features["min4"].cast('float'))
df_features = df_features.withColumn("max4", df_features["max4"].cast('float'))
df_features = df_features.withColumn("min5", df_features["min5"].cast('float'))
df_features = df_features.withColumn("max5", df_features["max5"].cast('float'))
df_features = df_features.withColumn("min6", df_features["min6"].cast('float'))
df_features = df_features.withColumn("max6", df_features["max6"].cast('float'))

# Checking the number of rows
print('total values:\n')
print(df_features.count())
# Ensuring no null values in final data
print('Null/NaN values: ')
df_features.select(
    [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in
     df_features.columns]).show()
# Checking the column data types
print(df_features.dtypes)
print('-' * 188)

# Creating the Formula for applying R on the data
formula = RFormula(formula='response ~ .', featuresCol='features',
                   labelCol='label')

# Transforming and fitting the data according to the formula
label_df = formula.fit(df_features).transform(df_features)
label_df.show(n=2)

# Splitting data into training and testing
train, test = label_df.randomSplit([0.75, 0.25], seed=42)

# Applying Random Forest on data
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
# Checking the predictions
predictions.select('birth_number_', 'date_acct_', 'gen', 'amount', 'payments',
                   'label', 'rawPrediction', 'prediction', 'probability').show(
    10)

# Checking the accuracy
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(
    evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
