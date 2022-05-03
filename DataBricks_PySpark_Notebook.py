# Databricks notebook source
# MAGIC %md Liam Neeson Programming for Big Data Assignment 2

# COMMAND ----------

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col, regexp_replace, expr, exp, count, when

# COMMAND ----------

house_price_path = "dbfs:/FileStore/shared_uploads/train.csv"

# COMMAND ----------

spark_session = SparkSession.builder.master("local[1]") \
    .appName("Housing_Regression_Assignment") \
    .getOrCreate()

# COMMAND ----------

spark_context = spark_session.sparkContext

# COMMAND ----------

house_price_data_frame = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(house_price_path)

# COMMAND ----------

# Show number of Rows and Columns in the DataFrame 
print((house_price_data_frame.count(), len(house_price_data_frame.columns)))
# Cache the training data will be accessing it alot 
house_price_data_frame.cache()

# COMMAND ----------

# Change the DataFrame to Pandas to Produce A correlation Matrix 
# Can do this in Spark doing something like: 
# house_price_data_frame.stat.corr("SalePrice","OverallQual")
# Could not figure out a way to do it for all columns at once so used a pandas for ease 
pandas_df = house_price_data_frame.toPandas()

# COMMAND ----------

# Correlation Matrix 
corr = pandas_df.corr()
corr[['SalePrice']].sort_values(by='SalePrice',ascending=False)

# COMMAND ----------

# MAGIC %md Check For Null Values 

# COMMAND ----------

null_count = house_price_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_data_frame.columns])
null_count.show(vertical=True)

# COMMAND ----------

house_price_data_frame.printSchema()

# COMMAND ----------

# Drop Columns With Majority Null / NA Values 
house_price_data_frame = house_price_data_frame.drop(col("Alley"))
house_price_data_frame = house_price_data_frame.drop(col("FireplaceQu"))
house_price_data_frame = house_price_data_frame.drop(col("PoolQC"))
house_price_data_frame = house_price_data_frame.drop(col("Fence"))
house_price_data_frame = house_price_data_frame.drop(col("MiscFeature"))

# COMMAND ----------

# Fill null values with 0 with LotFrontage
house_price_data_frame = house_price_data_frame.withColumn("LotFrontage", house_price_data_frame["LotFrontage"].cast('int'))
house_price_data_frame = house_price_data_frame.withColumn("MasVnrArea", house_price_data_frame["MasVnrArea"].cast('int'))
house_price_data_frame = house_price_data_frame.withColumn("GarageYrBlt", house_price_data_frame["GarageYrBlt"].cast('int'))
house_price_data_frame = house_price_data_frame.na.fill(0,["LotFrontage"])
house_price_data_frame = house_price_data_frame.na.fill(0,["MasVnrArea"])

# COMMAND ----------

# Tried to change all columsn with 'NA' to null this would have made it easier to change all values at once using something like this: 
#  house_price_data_frame = house_price_data_frame.na.fill(value="None",
#                                                         subset=["MasVnrType","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"])
# But it wouldn't work in the iteration I tried so stuck with the less efficent solution that worked 
# cols = [F.when(~F.col(x).isin("NULL", "NA", "NaN"), F.col(x)).alias(x)  for x in house_price_data_frame.columns]
# house_price_data_frame = house_price_data_frame.select(*cols)

# COMMAND ----------

# Change NA values within Basement Quality to None as these Properties don't have a Basement 
house_price_data_frame = house_price_data_frame.withColumn('BsmtQual', regexp_replace('BsmtQual', 'NA', "None"))

# COMMAND ----------

# Chnage NA values within Basement Quality to None as these Properties don't have a Basement 
house_price_data_frame = house_price_data_frame.withColumn('BsmtCond', regexp_replace('BsmtCond', 'NA', "None"))

# COMMAND ----------

# Change NA values within Basement Quality to None as these Properties don't have a Basement 
house_price_data_frame = house_price_data_frame.withColumn('GarageType', regexp_replace('garageType', 'NA', "None"))

# COMMAND ----------

# Chnage NA values with Masonry veneer type to None 
house_price_data_frame = house_price_data_frame.withColumn('MasVnrType', regexp_replace('MasVnrType', 'NA', "None"))

# COMMAND ----------

# Chnage NA values to 0 
house_price_data_frame = house_price_data_frame.na.fill(0,["MasVnrArea"])

# COMMAND ----------

# No Garage Year Built so chnage Value to None as no garage exists 
house_price_data_frame = house_price_data_frame.na.fill("None",["GarageYrBlt"])

# COMMAND ----------

# No Garage So No Finish so set to None 
house_price_data_frame = house_price_data_frame.withColumn('GarageFinish', regexp_replace('GarageFinish', 'NA', "None"))

# COMMAND ----------

# No garage exists so fill NA values for None 
house_price_data_frame = house_price_data_frame.withColumn('GarageQual', regexp_replace('GarageQual', 'NA', "None"))

# COMMAND ----------

# No garage exists 
house_price_data_frame = house_price_data_frame.withColumn('GarageCond', regexp_replace('GarageQual', 'NA', "None"))

# COMMAND ----------

# House has no basement 
house_price_data_frame = house_price_data_frame.withColumn('BsmtExposure', regexp_replace('BsmtExposure', 'NA', "None"))

# COMMAND ----------

# House has no basement 
house_price_data_frame = house_price_data_frame.withColumn('BsmtFinType1', regexp_replace('BsmtFinType1', 'NA', "None"))

# COMMAND ----------

# House has no basement 
house_price_data_frame = house_price_data_frame.withColumn('BsmtFinType2', regexp_replace('BsmtFinType1', 'NA', "None"))

# COMMAND ----------

# Every house has electricity so full with mode 
electrical_mode = house_price_data_frame.groupby("Electrical").count().orderBy("count", ascending=False).first()[0]
display(electrical_mode)
house_price_test_data_frame = house_price_data_frame.withColumn('Electrical', regexp_replace('Electrical', 'NA', electrical_mode))

# COMMAND ----------

# MAGIC %md Check for remaining NA / Nulls in Training Data  

# COMMAND ----------

# Verify removed all NA values 
null_count = house_price_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_data_frame.columns])
null_count.show(vertical=True)

# COMMAND ----------

# Verify no Null values 
null_count =  house_price_data_frame.select([count(when(col(c).isNull(),c)).alias(c) for c in house_price_data_frame.columns])
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

# Check Null Count in Test Data 
null_count = house_price_test_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_test_data_frame.columns])
null_count.show(vertical=True)

# COMMAND ----------

# Drop Same Columns from Test Data 
house_price_test_data_frame = house_price_test_data_frame.drop(col("Alley"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("FireplaceQu"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("PoolQC"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("Fence"))
house_price_test_data_frame = house_price_test_data_frame.drop(col("MiscFeature"))

# COMMAND ----------

house_price_test_data_frame.printSchema()

# COMMAND ----------

# After Reviewing the Schema needed to chnage some columns to type int
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

cols = ["LotFrontage", "MasVnrArea", "GarageYrBlt", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]
house_price_test_data_frame = house_price_test_data_frame.na.fill(value=0,subset=cols)

# COMMAND ----------

# Fill Zoning with Mode 
msz_mode = house_price_test_data_frame.groupby("MSZoning").count().orderBy("count", ascending=False).first()[0]
display(msz_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('MSZoning', regexp_replace('MSZoning', 'NA', msz_mode))

# COMMAND ----------

# Fill Utilites with Mode 
utilities_mode = house_price_test_data_frame.groupby("Utilities").count().orderBy("count", ascending=False).first()[0]
display(utilities_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('Utilities', regexp_replace('Utilities', 'NA', utilities_mode))

# COMMAND ----------

# Fill exterior1 adn exterior 2 with Mode 
ex1_mode = house_price_test_data_frame.groupby("Exterior1st").count().orderBy("count", ascending=False).first()[0]
display(ex1_mode)
ex2_mode = house_price_test_data_frame.groupby("Exterior2nd").count().orderBy("count", ascending=False).first()[0]
display(ex2_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('Exterior1st', regexp_replace('Exterior1st', 'NA', ex1_mode))
house_price_test_data_frame = house_price_test_data_frame.withColumn('Exterior2nd', regexp_replace('Exterior2nd', 'NA', ex1_mode))

# COMMAND ----------

# MasVnrType mode 
masVnrType_mode = house_price_test_data_frame.groupby("MasVnrType").count().orderBy("count", ascending=False).first()[0]
display(masVnrType_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('MasVnrType', regexp_replace('MasVnrType', 'NA', masVnrType_mode))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtQual', regexp_replace('BsmtQual', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtCond', regexp_replace('BsmtCond', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtExposure', regexp_replace('BsmtExposure', 'NA', "None"))

# COMMAND ----------

house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtFinType1', regexp_replace('BsmtFinType1', 'NA', "None"))
house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtFinType2', regexp_replace('BsmtFinType2', 'NA', "None"))

# COMMAND ----------

bsmtFinSF1_mode = house_price_test_data_frame.groupby("BsmtFinSF1").count().orderBy("count", ascending=False).first()[0]
display(bsmtFinSF1_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtFinSF1', regexp_replace('BsmtFinSF1', 'NA', "None"))

# COMMAND ----------

# Fill functional with mode 
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

bsmtUnfSF_mode = house_price_test_data_frame.groupby("BsmtUnfSF").count().orderBy("count", ascending=False).first()[0]
display(bsmtUnfSF_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('BsmtUnfSF', regexp_replace('BsmtUnfSF', 'NA', bsmtUnfSF_mode))

# COMMAND ----------

kitch_qual_mode = house_price_test_data_frame.groupby("KitchenQual").count().orderBy("count", ascending=False).first()[0]
display(kitch_qual_mode)
house_price_test_data_frame = house_price_test_data_frame.withColumn('KitchenQual', regexp_replace('KitchenQual', 'NA', kitch_qual_mode))

# COMMAND ----------

null_count = house_price_data_frame.select([count(when(col(c).contains('NA'), c)).alias(c) for c in house_price_data_frame.columns])

null_count.show(vertical=True)

# COMMAND ----------

house_price_data_frame.printSchema()

# COMMAND ----------

# MAGIC %md Added some extra columns to try and improve the accuracy of model

# COMMAND ----------

# all the features that correlate highly to sales price to their own column 
house_price_data_frame = house_price_data_frame.withColumn('HighCorr',expr("OverallQual * GrLivArea * GarageCars * GarageArea"))
house_price_test_data_frame = house_price_test_data_frame.withColumn('HighCorr',expr("OverallQual * GrLivArea * GarageCars * GarageArea"))

# Total SQFT of All the building 
house_price_data_frame = house_price_data_frame.withColumn('TotalSF',expr("TotalBsmtSF + 1stFlrSF + 2ndFlrSF"))
house_price_test_data_frame = house_price_test_data_frame.withColumn('TotalSF',expr("TotalBsmtSF + 1stFlrSF + 2ndFlrSF"))

# Sum Square foot of the main part of the house 
house_price_data_frame = house_price_data_frame.withColumn('MainSF',expr("1stFlrSF + 2ndFlrSF"))
house_price_test_data_frame = house_price_test_data_frame.withColumn('MainSF',expr("TotalBsmtSF + 1stFlrSF + 2ndFlrSF"))

# Total Number of Bathroom in the house 
house_price_data_frame = house_price_data_frame.withColumn('TotalBathroom',expr("FullBath + HalfBath"))
house_price_test_data_frame = house_price_test_data_frame.withColumn('TotalBathroom',expr("FullBath + HalfBath"))

# COMMAND ----------

# Confirm that these new columns were added
# house_price_data_frame.printSchema()

# COMMAND ----------

# MAGIC %md Data Visualisation to Remove Outliers: Removed outliers from the top 6 Highly Correlated Categories  

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# changed to pandas for easier visualistions also made it easier to remove values from a dataframe conditionaly
# tried implementing this using the pyspark sql using the where condition but didn't work below the pandas commands 
# to remove outliers I included what they would look like using pyspark 
# Also made more sense to remove them using pandas to confirm using the visualisations that they were indeed removed. (But mainly because didn't work when I tried and ran out of time to fix ) 
pd_train = house_price_data_frame.toPandas()
pd_test = house_price_test_data_frame.toPandas()

# COMMAND ----------


fig, axes = plt.subplots(1, 2, figsize=(25,15))
axes[0].set_xlim(0, 12)
axes[1].set_xlim(-1, 6000)

# OverallQual vs SalePrice 
sns.scatterplot(data=pd_train, ax=axes[0], x='OverallQual', y='SalePrice').grid()
axes[0].set_title('OverallQual vs SalePrice')

# GrLivArea vs SalePrice
sns.scatterplot(data=pd_train, ax=axes[1], x='GrLivArea', y='SalePrice').grid()
axes[1].set_title('GrLivArea vs SalePrice')

# COMMAND ----------

pd_train = pd_train.drop(pd_train[(pd_train['SalePrice']>600000)].index)
# house_price_data_frame = house_price_data_frame.where("SalePrice > 6000000") 
pd_train = pd_train.drop(pd_train[(pd_train['SalePrice']<200000) & (pd_train['OverallQual']>9)].index)
# house_price_data_frame = house_price_data_frame.where("SalePrice > 6000000 AND OverallQual > 9") 
pd_train = pd_train.drop(pd_train[(pd_train['GrLivArea']>4000) & (pd_train['SalePrice']<500000)].index)
# house_price_data_frame = house_price_data_frame.where("GrLivArea > 4000 AND OverallQual > 9") 

# COMMAND ----------

# Verify the outliers were removed 
fig, axes = plt.subplots(1, 2, figsize=(25,15))
axes[0].set_xlim(0, 12)
axes[1].set_xlim(-1, 6000)

sns.scatterplot(data=pd_train, ax=axes[0], x='OverallQual', y='SalePrice').grid()
axes[0].set_title('OverallQual vs SalePrice')
sns.scatterplot(data=pd_train, ax=axes[1], x='GrLivArea', y='SalePrice').grid()
axes[1].set_title('GrLivArea vs SalePrice')

# COMMAND ----------

# MAGIC %md Garage Cars & Garage Area 

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(25,15))
axes[0].set_xlim(-1, 6)
axes[1].set_xlim(-100, 2000)

# GarageCars vs SalePrice
sns.scatterplot(data=pd_train, ax=axes[0], x='GarageCars', y='SalePrice').grid()
axes[0].set_title('GarageCars vs SalePrice')

# GarageArea vs SalePrice
sns.scatterplot(data=pd_train, ax=axes[1], x='GarageArea', y='SalePrice').grid()
axes[1].set_title('GarageArea vs SalePrice')

# COMMAND ----------

pd_train = pd_train.drop(pd_train[(pd_train['GarageCars']>3.5) | (pd_train['SalePrice']>500000)].index)
# house_price_data_frame = house_price_data_frame.where("GarageCars > 3.5 OR SalePrice > 500000") 
pd_train = pd_train.drop(pd_train[(pd_train['GarageArea']>1100)].index)
# house_price_data_frame = house_price_data_frame.where("GarageArea > 1100") 

# COMMAND ----------

# Verify 
fig, axes = plt.subplots(1, 2, figsize=(25,15))
axes[0].set_xlim(-1, 4)
axes[1].set_xlim(-100, 1500)

# GarageCars vs SalePrice
sns.scatterplot(data=pd_train, ax=axes[0], x='GarageCars', y='SalePrice').grid()
axes[0].set_title('GarageCars vs SalePrice')

# GarageArea vs SalePrice
sns.scatterplot(data=pd_train, ax=axes[1], x='GarageArea', y='SalePrice').grid()
axes[1].set_title('GarageArea vs SalePrice')

# COMMAND ----------

# MAGIC %md Total Basement SF & 1st Floor SF 

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(25,15))
axes[0].set_xlim(-1000, 6000)
axes[1].set_xlim(0, 6000)

sns.scatterplot(data=pd_train, ax=axes[0], x='TotalBsmtSF', y='SalePrice').grid()
axes[0].set_title('TotalBsmtSF vs SalePrice')
sns.scatterplot(data=pd_train, ax=axes[1], x='1stFlrSF', y='SalePrice').grid()
axes[1].set_title('1stFlrSF vs SalePrice')

# COMMAND ----------

pd_train = pd_train.drop(pd_train[(pd_train['TotalBsmtSF']>2500) | (pd_train['SalePrice']>450000)].index)
# house_price_data_frame = house_price_data_frame.where("TotalBsmtSF > 2500 OR SalePrice > 450000") 
pd_train = pd_train.drop(pd_train[(pd_train['1stFlrSF']>2500) | (pd_train['SalePrice']>450000)].index)
# house_price_data_frame = house_price_data_frame.where("1stFlrSF > 2500 OR SalePrice > 450000") 

# COMMAND ----------

#  Verify 
fig, axes = plt.subplots(1, 2, figsize=(25,15))
axes[0].set_xlim(-1000, 3000)
axes[1].set_xlim(0, 3000)

sns.scatterplot(data=pd_train, ax=axes[0], x='TotalBsmtSF', y='SalePrice').grid()
axes[0].set_title('TotalBsmtSF vs SalePrice')
sns.scatterplot(data=pd_train, ax=axes[1], x='1stFlrSF', y='SalePrice').grid()
axes[1].set_title('1stFlrSF vs SalePrice')

# COMMAND ----------

train_cols = list(pd_train.columns)
train_cols.remove('SalePrice')

# COMMAND ----------

# Change back to spark DF 
train_df = spark_session.createDataFrame(pd_train)
test_df = spark_session.createDataFrame(pd_test)

# COMMAND ----------

train_cols = train_df.columns
train_cols.remove('SalePrice')
print(train_cols)
test_df = test_df.select(train_cols)

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType

# May need to change DataTypes when convertin back to spark DataFrame
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

# Double check the change of data type in train data frame 
train_df = train_df.withColumn("MasVnrArea", train_df["MasVnrArea"].cast(IntegerType()))
train_df = train_df.na.fill(0,["MasVnrArea"])
train_df = train_df.withColumn("LotFrontage", train_df["LotFrontage"].cast(IntegerType()))
train_df = train_df.na.fill(0,["LotFrontage"])
train_df = train_df.withColumn("GarageYrBlt", train_df["GarageYrBlt"].cast(IntegerType()))
train_df = train_df.na.fill(0,["GarageYrBlt"])

# COMMAND ----------

categorical_columns_train = []

# Check to see which columns are categorical as they will need an index  
for col, dtype in train_df.dtypes:
    if dtype == 'string':
        categorical_columns_train.append(col)
        
print(categorical_columns_train)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# Create a Index Value for all Categorical Values within a Colum 
index_train = [StringIndexer(inputCol=column, outputCol=column+'Index', handleInvalid='keep').fit(train_df) for column in categorical_columns_train]

pipeline_train = Pipeline(stages=index_train)
train_index = pipeline_train.fit(train_df).transform(train_df)
# display(train_index)

# COMMAND ----------

categorical_columns_test = []

# Check to see which columns are categorical as they will need an index in the test data 
for col, dtype in test_df.dtypes:
    if dtype == 'string':
        categorical_columns_test.append(col)

# COMMAND ----------

# Create a Index Value for all Categorical Values within the test data
indexer_test = [StringIndexer(inputCol=column, outputCol=column+'Index', handleInvalid='keep').fit(test_df) for column in categorical_columns_test]

pipeline_train = Pipeline(stages=indexer_test)
test_index = pipeline_train.fit(test_df).transform(test_df)

# COMMAND ----------

def get_dtype(df,colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]

numerical_columns_train = []
for col in train_index.columns:
    if get_dtype(train_index,col) != 'string':
        numerical_columns_train.append(str(col))
        
numerical_columns_test = []
for col in test_index.columns:
    if get_dtype(test_index,col) != 'string':
        numerical_columns_test.append(str(col))

train_index = train_index.select(numerical_columns_train)
test_index = test_index.select(numerical_columns_test)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Add all of the Numerical Features to a singular column called features so that it can be added to the Model
vectorAssembler_train = VectorAssembler(inputCols = train_index.drop("SalePrice").columns, outputCol = 'features').setHandleInvalid("keep")

train_vector = vectorAssembler_train.transform(train_index)

# COMMAND ----------

# Add all of the Numerical Features to a singular column called features so that it can be added to the Model
vectorAssembler_test= VectorAssembler(inputCols = test_index.columns, outputCol = 'features').setHandleInvalid("keep")

test_vector = vectorAssembler_test.transform(test_index)

# COMMAND ----------

from pyspark.sql.functions import lit
# Create a Column SalePrice in the test 
test_vector = test_vector.withColumn("SalePrice", lit(0))

# COMMAND ----------

# split the data into a training and test set 
train_set, test_set = train_vector.randomSplit([0.7, 0.3])

# COMMAND ----------

# MAGIC %md Linear Regression 

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='SalePrice', maxIter=10, regParam=0.8, elasticNetParam=0.1)
lr_model = lr.fit(train_set)

lr_predictions = lr_model.transform(test_set)
display(lr_predictions.select("prediction","SalePrice","features"))

# COMMAND ----------

# Add the Test Vector Data to the Model
lr_predictions_test_df = lr_model.transform(test_vector)
lr_pred = lr_predictions_test_df.select("Id","prediction")
# Rename Columns ot fit the submission 
lr_pred = lr_pred.withColumnRenamed("prediction","SalePrice")


lr_pred = lr_pred.withColumn("Id", lr_pred["Id"].cast(IntegerType()))
lr_pred = lr_pred.withColumn("SalePrice", lr_pred["SalePrice"].cast(FloatType()))

# COMMAND ----------

# Download Results for Linear Regression 

display(lr_pred)

# COMMAND ----------

# MAGIC %md Decision Tree

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(featuresCol = 'features', labelCol='SalePrice', maxBins=32, maxDepth=20)
dt_model = dt.fit(train_set)

dt_predictions = dt_model.transform(test_set)
display(dt_predictions.select("prediction","SalePrice","features"))

# COMMAND ----------

# Add the Test Vector Data to the Model
dt_predictions_test_df = dt_model.transform(test_vector)
dt_pred = dt_predictions_test_df.select("Id","prediction")
dt_pred = dt_pred.withColumnRenamed("prediction","SalePrice")

dt_pred = dt_pred.withColumn("Id", dt_pred["Id"].cast(IntegerType()))
dt_pred = dt_pred.withColumn("SalePrice", dt_pred["SalePrice"].cast(FloatType()))

# COMMAND ----------

# Download Decision Tree Reulsts 

display(dt_pred)

# COMMAND ----------

# MAGIC %md Random Forest Regressor 

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol = 'features', labelCol='SalePrice',maxDepth=20, minInstancesPerNode=2, bootstrap=True)
rf_model = rf.fit(train_set)

rf_predictions = rf_model.transform(test_set)
display(rf_predictions.select("prediction","SalePrice","features"))

# COMMAND ----------

rf_predictions_teset_df = rf_model.transform(test_vector)
rf_pred = rf_predictions_teset_df.select("Id","prediction")
rf_pred = rf_pred.withColumnRenamed("prediction","SalePrice")


rf_pred = rf_pred.withColumn("Id", rf_pred["Id"].cast(IntegerType()))
rf_pred = rf_pred.withColumn("SalePrice", rf_pred["SalePrice"].cast(FloatType()))

# COMMAND ----------

# Download Random Forest Reulsts 

display(rf_pred)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="r2") 
dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="r2") 
rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="r2")

linear_r2 = lr_evaluator.evaluate(lr_predictions) * 100
decision_tree_r2 = dt_evaluator.evaluate(dt_predictions) * 100
random_forest_r2 = rf_evaluator.evaluate(rf_predictions) * 100

print(f"R Squared for Logistic Regression is: {linear_r2}%")
print(f"R Squared for Decision Tree is: {decision_tree_r2}%")
print(f"R Squared for Random Forest is: {random_forest_r2}%")
