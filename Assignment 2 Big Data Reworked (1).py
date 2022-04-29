# Databricks notebook source
# MAGIC %md Liam Neeson Programming for Big Data Assignment 2

# COMMAND ----------

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

house_price_path = "dbfs:/FileStore/shared_uploads/train.csv"

# COMMAND ----------

house_price_data_frame = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(house_price_path)

# COMMAND ----------

spark_session = SparkSession.builder.master("local[2]").appName("HousingRegression").getOrCreate()

# COMMAND ----------

spark_context = spark_session.sparkContext

# COMMAND ----------

pandas_df = house_price_data_frame.toPandas()
na_cols = pandas_df.columns[pandas_df.isna().any()].tolist()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew 

sns.distplot(pandas_df['SalePrice'] , fit=norm);

# parameters
(mu, sigma) = norm.fit(pandas_df['SalePrice'])

plt.suptitle('Normal distribution with mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# COMMAND ----------

corr = pandas_df.corr()
corr[['SalePrice']].sort_values(by='SalePrice',ascending=False).style.background_gradient(cmap='viridis', axis=None)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
axes[0].set_xlim(0,10)

sns.scatterplot(data=pandas_df, ax=axes[0], x='OverallQual', y='SalePrice')
axes[0].set_title('OverallQual vs SalePrice')
sns.scatterplot(data=pandas_df, ax=axes[1], x='GarageCars', y='SalePrice')
axes[1].set_title('GarageCars vs SalePrice')

# COMMAND ----------

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
axes[0].set_xlim(0, 6000)

sns.scatterplot(data=pandas_df, ax=axes[0], x='GrLivArea', y='SalePrice')
axes[0].set_title('GrLivArea vs SalePrice')
sns.scatterplot(data=pandas_df, ax=axes[1], x='GarageArea', y='SalePrice')
axes[1].set_title('GarageArea vs SalePrice')

# COMMAND ----------

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
axes[0].set_xlim(0, 6000)

sns.scatterplot(data=pandas_df, ax=axes[0], x='TotalBsmtSF', y='SalePrice')
axes[0].set_title('TotalBsmtSF vs SalePrice')
sns.scatterplot(data=pandas_df, ax=axes[1], x='1stFlrSF', y='SalePrice')
axes[1].set_title('1stFlrSF vs SalePrice')

# COMMAND ----------

# MAGIC %md Check For Null Values 

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count
null_count = house_price_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_data_frame.columns])

display(null_count)

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn("LotFrontage", house_price_data_frame["LotFrontage"].cast('int'))
house_price_data_frame = house_price_data_frame.withColumn("MasVnrArea", house_price_data_frame["MasVnrArea"].cast('int'))
house_price_data_frame = house_price_data_frame.withColumn("GarageYrBlt", house_price_data_frame["GarageYrBlt"].cast('int'))

# COMMAND ----------

house_price_data_frame.printSchema()

# COMMAND ----------

house_price_data_frame = house_price_data_frame.drop(col("Alley"))
house_price_data_frame = house_price_data_frame.drop(col("FireplaceQu"))
house_price_data_frame = house_price_data_frame.drop(col("PoolQC"))
house_price_data_frame = house_price_data_frame.drop(col("Fence"))
house_price_data_frame = house_price_data_frame.drop(col("MiscFeature"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.na.fill(0,["LotFrontage"])

# COMMAND ----------

house_price_data_frame.printSchema()

# COMMAND ----------

from pyspark.sql.functions import regexp_replace
house_price_data_frame = house_price_data_frame.withColumn('BsmtQual', regexp_replace('BsmtQual', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('BsmtCond', regexp_replace('BsmtCond', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('GarageType', regexp_replace('garageType', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('MasVnrType', regexp_replace('MasVnrType', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('MasVnrArea', regexp_replace('MasVnrArea', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.na.fill("None",["GarageYrBlt"])

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('GarageFinish', regexp_replace('GarageFinish', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('GarageQual', regexp_replace('GarageQual', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('GarageCond', regexp_replace('GarageQual', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('BsmtExposure', regexp_replace('BsmtExposure', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('BsmtFinType1', regexp_replace('BsmtFinType1', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.withColumn('BsmtFinType2', regexp_replace('BsmtFinType1', 'NA', "None"))

# COMMAND ----------

house_price_data_frame = house_price_data_frame.filter(house_price_data_frame.Electrical != 'NA')

# COMMAND ----------

null_count = house_price_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

# MAGIC %md Load in the Test Data 

# COMMAND ----------

house_price_test_path = "dbfs:/FileStore/shared_uploads/test.csv"

# COMMAND ----------

house_price_test_data_frame = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(house_price_test_path)

# COMMAND ----------

null_count = house_price_test_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_test_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.drop(col("Alley"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("FireplaceQu"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("PoolQC"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("Fence"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("MiscFeature"))

# COMMAND ----------

house_price_test_data_frame.printSchema()

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn("LotFrontage", house_price_test_data_frame["LotFrontage"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("MasVnrArea", house_price_test_data_frame["MasVnrArea"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("GarageYrBlt", house_price_test_data_frame["GarageYrBlt"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("BsmtFinSF1", house_price_test_data_frame["BsmtFinSF1"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("BsmtFinSF2", house_price_test_data_frame["BsmtFinSF2"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("TotalBsmtSF", house_price_test_data_frame["TotalBsmtSF"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("BsmtFullBath", house_price_test_data_frame["BsmtFullBath"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("BsmtHalfBath", house_price_test_data_frame["BsmtHalfBath"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("GarageCars", house_price_test_data_frame["GarageCars"].cast('int'))
house_price_test_data_frame = house_price_test_data_frame.withColumn("GarageArea", house_price_test_data_frame["GarageArea"].cast('int'))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.na.fill(value=0,subset=["LotFrontage", "MasVnrArea", "GarageYrBlt", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"])

# COMMAND ----------

msz_mode = house_price_test_data_frame.groupby("MSZoning").count().orderBy("count", ascending=False).first()[0]
display(msz_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('MSZoning', regexp_replace('MSZoning', 'NA', msz_mode))

# COMMAND ----------

utilities_mode = house_price_test_data_frame.groupby("Utilities").count().orderBy("count", ascending=False).first()[0]
display(utilities_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('Utilities', regexp_replace('Utilities', 'NA', utilities_mode))

# COMMAND ----------

ex1_mode = house_price_test_data_frame.groupby("Exterior1st").count().orderBy("count", ascending=False).first()[0]
display(ex1_mode)
ex2_mode = house_price_test_data_frame.groupby("Exterior2nd").count().orderBy("count", ascending=False).first()[0]
display(ex2_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('Exterior1st', regexp_replace('Exterior1st', 'NA', ex1_mode))
house_price_test_data_frame = house_price_test_data_frame.withColumn('Exterior2nd', regexp_replace('Exterior2nd', 'NA', ex1_mode))

# COMMAND ----------

masVnrType_mode = house_price_test_data_frame.groupby("MasVnrType").count().orderBy("count", ascending=False).first()[0]
display(masVnrType_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('MasVnrType', regexp_replace('MasVnrType', 'NA', masVnrType_mode))

# COMMAND ----------

null_count = house_price_test_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_test_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtQual', regexp_replace('BsmtQual', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtCond', regexp_replace('BsmtCond', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtExposure', regexp_replace('BsmtExposure', 'NA', "None"))

# COMMAND ----------

null_count = house_price_test_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_test_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtFinType1', regexp_replace('BsmtFinType1', 'NA', "None"))
house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtFinType2', regexp_replace('BsmtFinType2', 'NA', "None"))

# COMMAND ----------

bsmtFinSF1_mode = house_price_test_data_frame.groupby("BsmtFinSF1").count().orderBy("count", ascending=False).first()[0]
display(bsmtFinSF1_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtFinSF1', regexp_replace('BsmtFinSF1', 'NA', "None"))

# COMMAND ----------

functional_mode = house_price_test_data_frame.groupby("Functional").count().orderBy("count", ascending=False).first()[0]
display(functional_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('Functional', regexp_replace('Functional', 'NA', functional_mode))

# COMMAND ----------

saleType_mode = house_price_test_data_frame.groupby("SaleType").count().orderBy("count", ascending=False).first()[0]
display(saleType_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('SaleType', regexp_replace('SaleType', 'NA', saleType_mode))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('GarageFinish', regexp_replace('GarageFinish', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('GarageType', regexp_replace('GarageType', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('GarageQual', regexp_replace('GarageQual', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('GarageCond', regexp_replace('GarageCond', 'NA', "None"))

# COMMAND ----------

null_count = house_price_test_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_test_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

bsmtUnfSF_mode = house_price_test_data_frame.groupby("BsmtUnfSF").count().orderBy("count", ascending=False).first()[0]
display(bsmtUnfSF_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtUnfSF', regexp_replace('BsmtUnfSF', 'NA', bsmtUnfSF_mode))

# COMMAND ----------

kitch_qual_mode = house_price_test_data_frame.groupby("KitchenQual").count().orderBy("count", ascending=False).first()[0]
display(kitch_qual_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('KitchenQual', regexp_replace('KitchenQual', 'NA', kitch_qual_mode))

# COMMAND ----------

null_count = house_price_test_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_test_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

null_count = house_price_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

pd_train = house_price_data_frame.toPandas()
pd_test = house_price_test_data_frame.toPandas()

# COMMAND ----------

pd_train['New'] = pd_train['OverallQual'] * pd_train['GarageArea'] * pd_train['GrLivArea']
pd_test['New'] = pd_test['OverallQual'] * pd_test['GarageArea'] * pd_test['GrLivArea']

# COMMAND ----------

pd_train = pd_train.drop(pd_train[(pd_train['GrLivArea']>4500) 
                                & (pd_train['SalePrice']<300000)].index)

pd_train = pd_train.drop(pd_train[(pd_train['GrLivArea']>5500) 
                                | (pd_train['SalePrice']>500000)].index)

pd_train = pd_train.drop(pd_train[pd_train['GarageArea']>1100].index)

# COMMAND ----------

train_cols = list(pd_train.columns)
train_cols.remove('SalePrice')

# COMMAND ----------

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

train_df = spark.createDataFrame(pd_train)
test_df = spark.createDataFrame(pd_test)

# COMMAND ----------

train_df = train_df.select([c for c in train_df.columns if c not in na_cols])
train_cols = train_df.columns
train_cols.remove('SalePrice')
test_df = test_df.select(train_cols)

# COMMAND ----------



from pyspark.sql.types import IntegerType

# As PySpark DFs can be finicky, sometimes your have to explicitly cast certain data types to columns

test_df = test_df.withColumn("BsmtFinSF1", test_df["BsmtFinSF1"].cast(IntegerType()))
test_df = test_df.withColumn("BsmtFinSF2", test_df["BsmtFinSF2"].cast(IntegerType()))
test_df = test_df.withColumn("BsmtUnfSF", test_df["BsmtUnfSF"].cast(IntegerType()))
test_df = test_df.withColumn("TotalBsmtSF", test_df["TotalBsmtSF"].cast(IntegerType()))
test_df = test_df.withColumn("BsmtFullBath", test_df["BsmtFullBath"].cast(IntegerType()))
test_df = test_df.withColumn("BsmtHalfBath", test_df["BsmtHalfBath"].cast(IntegerType()))
test_df = test_df.withColumn("GarageCars", test_df["GarageCars"].cast(IntegerType()))
test_df = test_df.withColumn("GarageArea", test_df["GarageArea"].cast(IntegerType()))

train_df.printSchema()

# COMMAND ----------

train_df = train_df.withColumn("MasVnrArea", train_df["MasVnrArea"].cast('int'))
train_df= train_df.na.fill(0,["MasVnrArea"])

# COMMAND ----------

train_string_columns = []

for col, dtype in train_df.dtypes:
    if dtype == 'string':
        train_string_columns.append(col)
        
print(train_string_columns)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=column, outputCol=column+'_index', handleInvalid='keep').fit(train_df) for column in train_string_columns]


pipeline = Pipeline(stages=indexers)
train_indexed = pipeline.fit(train_df).transform(train_df)

# COMMAND ----------

print(len(train_indexed.columns))

# COMMAND ----------

test_string_columns = []

for col, dtype in test_df.dtypes:
    if dtype == 'string':
        test_string_columns.append(col)

# COMMAND ----------

indexers2 = [StringIndexer(inputCol=column, outputCol=column+'_index', handleInvalid='keep').fit(test_df) for column in test_string_columns]

pipeline2 = Pipeline(stages=indexers2)
test_indexed = pipeline2.fit(test_df).transform(test_df)

# COMMAND ----------

print(len(test_indexed.columns))

# COMMAND ----------

def get_dtype(df,colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]

num_cols_train = []
for col in train_indexed.columns:
    if get_dtype(train_indexed,col) != 'string':
        num_cols_train.append(str(col))
        
num_cols_test = []
for col in test_indexed.columns:
    if get_dtype(test_indexed,col) != 'string':
        num_cols_test.append(str(col))

train_indexed = train_indexed.select(num_cols_train)
test_indexed = test_indexed.select(num_cols_test)

# COMMAND ----------

print(len(train_indexed.columns))
print(len(test_indexed.columns))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = train_indexed.drop("SalePrice").columns, outputCol = 'features').setHandleInvalid("keep")

train_vector = vectorAssembler.transform(train_indexed)

# COMMAND ----------

vectorAssembler2 = VectorAssembler(inputCols = test_indexed.columns, outputCol = 'features').setHandleInvalid("keep")

test_vector = vectorAssembler2.transform(test_indexed)

# COMMAND ----------

from pyspark.sql.functions import lit

test_vector = test_vector.withColumn("SalePrice", lit(0))

# COMMAND ----------

splits = train_vector.randomSplit([0.7, 0.3])
train = splits[0]
val = splits[1]

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='SalePrice', maxIter=10, 
                      regParam=0.8, elasticNetParam=0.1) # It is always a good idea to play with hyperparameters.
lr_model = lr.fit(train)

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(val)
lr_predictions.select("prediction","SalePrice","features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="SalePrice",metricName="r2")
print("R Squared (R2) on val data = %g" % lr_evaluator.evaluate(lr_predictions))

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol = 'features', labelCol='SalePrice', 
                           maxDepth=20, 
                           minInstancesPerNode=2,
                           bootstrap=True
                          )
rf_model = rf.fit(train)

rf_predictions = rf_model.transform(val)
rf_predictions.select("prediction","SalePrice","features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator
rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="SalePrice",metricName="r2")
print("R Squared (R2) on val data = %g" % rf_evaluator.evaluate(rf_predictions))

# COMMAND ----------

rf_predictions2 = rf_model.transform(test_vector)
#rf_predictions2.printSchema()
pred = rf_predictions2.select("Id","prediction")
pred = pred.withColumnRenamed("prediction","SalePrice")

from pyspark.sql.types import FloatType, IntegerType

#pred.printSchema()
pred = pred.withColumn("Id", pred["Id"].cast(IntegerType()))
pred = pred.withColumn("SalePrice", pred["SalePrice"].cast(FloatType()))

# COMMAND ----------

display(pred)
