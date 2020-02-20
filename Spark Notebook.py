!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://apache.osuosl.org/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
!tar xf spark-2.4.4-bin-hadoop2.7.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"

import findspark
findspark.init()

# Load additional libraries 
import pyspark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.ml.stat import Correlation, KolmogorovSmirnovTest
import matplotlib.pyplot as plt
from pyspark.sql.functions import col,when,count
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro 
import time 

# Load files 
from google.colab import files
files.upload()


spark = SparkSession.builder.master('local[*]').appName('Spark_Coursework').getOrCreate()

spark_crimes = spark.read.csv('.../formatted_crimes.csv', inferSchema=True, header=True, encoding = "ISO-8859-1")

# Observe dimensions of the data:
print(
    'Rows:', data.count(), 'Columns:', len(data.columns)
)

# Take a look at the data 
data.show(5)

# Observe data types for the DataFrame
print('Data types:')
data.dtypes

# Observe Schema
print('Schema:')
data.printSchema()

# Firstly I will drop some columns - this will involve creating a new dataframe which is a subset of the previous one
data = data.select('OFFENSE_CODE_GROUP','DISTRICT','SHOOTING','YEAR','MONTH','DAY_OF_WEEK','HOUR',
                  'Lat','Long','DATE')
data.show(5)

# Get null value counts
print('Columns and respective null value counts:')
data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).show(5)

# Since the dataset is large I will simply drop the na rows for simplicity
data = data.dropna()
# New counts 
print(
    'Rows:', data.count(), 'Columns:', len(data.columns)
)

# Unique value counts
for i in data.columns:
    print(i+':',data.select(i).distinct().count())

# Create Time Series of Number of Crimes by Date
crime_series = data.groupBy('DATE').count()
print('Number of days:',crime_series.count())
crime_series.show(5)

# Need to sort the series by date - so the entries are in chronological order
crimes_df = crime_series.orderBy(crime_series.DATE.asc())
crimes_df.show(5)

# Show some summary statistics about the time series here
crimes_df.describe().show()

# I will create variables containing the mean and standard deviation since they will likely come in use later
mu = 251.97
sigma = 30.52

# PySpark currently has no plotting functionality therefore must use the built in toPandas method
# to create a pandas version of my data which I can then plot.
df_pd = crimes_df.toPandas()
df_pd.head()

# Plot the time series 
plt.figure(figsize=(15,10))
plt.plot(df_pd['DATE'], df_pd['count'])
plt.title('Time Series Plot of Crimes Each Day')
plt.ylabel('Number of Crimes')
plt.xlabel('Date')

# Plot cumulative crimes
plt.figure(figsize=(15,10))
plt.plot(df_pd['DATE'], np.cumsum(df_pd['count']))
plt.title('Cumulative Number of Crimes')
plt.ylabel('Number of Crimes')
plt.xlabel('Date')

# Plot the distribution via histogram 
plt.figure(figsize=(10,10))
plt.hist(df_pd['count'],bins=50,density=True)
plt.title('Histogram of Crimes per Day')
plt.ylabel('Number of Crimes')
plt.xlabel('Date')
plt.vlines(x=mu-sigma,ymin=0,ymax=0.016,label='mu-sigma',linestyles='dashed')
plt.vlines(x=mu+sigma,ymin=0,ymax=0.016,label='mu+sigma',linestyles='dashed')
plt.vlines(x=mu,ymin=0,ymax=0.016)

# Test for normality of data
print('P-value for normality test:',shapiro(df_pd['count'])[1])

# Dataset for regression to be plotted 
regress = pd.DataFrame()
regress['x'] = df_pd['count'].shift(1)
regress['b'] = pd.Series([1] * len(df_pd))
regress['y'] = df_pd['count']
regress.dropna(inplace=True)
regress.head()

# Scatter plot of t-1 vs t values
plt.figure(figsize=(7,7))
plt.title('Scatter plot of values at time t-1 against values at time t')
plt.xlabel('t-1')
plt.ylabel('t')
plt.scatter(regress['x1'],regress['y'])

# Change to spark df
regression_data = spark.createDataFrame(regress)
regression_data.show(10)

# Recode all the regression data to 'double' types 
regression_data = regression_data.withColumn('x',regression_data['x'].cast(DoubleType()))
regression_data = regression_data.withColumn('b',regression_data['b'].cast(DoubleType()))
regression_data = regression_data.withColumn('y',regression_data['y'].cast(DoubleType()))
print('Data types:')
regression_data.dtypes

# Use VectorAssembler
inputcols = ['x','b']
assembler = VectorAssembler(inputCols= inputcols,
                            outputCol = "predictors")
predictors = assembler.transform(regression_data)
predictors.columns

model_data = predictors.select("predictors", "y")
model_data.show(5,truncate=False)

# Split data and create, fit and predict model 
train_data,test_data = model_data.randomSplit([0.8,0.2])
lr = LinearRegression(
    featuresCol = 'predictors', 
    labelCol = 'y')
lrModel = lr.fit(train_data)
pred = lrModel.evaluate(test_data)

print('B0 coefficient:',lrModel.coefficients[1],', B1 coefficient:',lrModel.coefficients[0])

# Show predictions alongside y values
pred.predictions.show(10)

# Evaluate the regression 
eval = RegressionEvaluator(
    labelCol="y", 
    predictionCol="prediction", 
    metricName="mse")
mse = eval.evaluate(pred.predictions, {eval.metricName: "mse"})
r2 = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
print('MSE:',mse,', R2:',r2)

# Compared to sklearn 
train, test = train_test_split(regress,test_size=0.2)
lin_model = LinearRegression()
# Sklearn time taken 
s1 = time.time()
lin_model.fit(X=train.iloc[:,[0,1]],y=train['y'])
preds = lin_model.predict(test.iloc[:,[0,1]])
s2 = time.time()
print('Time taken:',s2-s1,'seconds.')

# PySpark time taken 
s1 = time.time()
lrModel = lr.fit(train_data)
pred = lrModel.evaluate(test_data)
s2 = time.time()
print('Time taken:',s2-s1,'seconds')
