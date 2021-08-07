from pyspark import SparkContext
from pyspark.ml import feature
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml import Pipeline

import numpy as np
import boto3

def train(df, year, first_half):

   if first_half: # first half of year
      train = df.filter(df.year==year).filter(df.month <= 6)
      test = df.filter(df.year==year).filter(df.month == 7)
   else: # second half of year
      train = df.filter(df.year==year).filter(df.month > 6)
      test = df.filter(df.year==year+1).filter(df.month == 1)

   # model training
   rf = RandomForestRegressor(featuresCol="features", labelCol="profit")
   model = rf.fit(train)

   # evaluation
   predictions = model.transform(test)
   predictions = predictions.withColumn("APE", 100.0*abs((predictions.prediction-predictions.profit)/predictions.profit))
   mape = predictions.agg(mean("APE").alias("MAPE")).collect()[0]; mape = mape["MAPE"]
   result = "First Half of Year %.2f, " % year +  "MAPE: %.2f\n" % mape; print(result)

   return mape, result


if __name__ == "__main__":
   sc = SparkContext(appName="exercise1"); sc.setLogLevel("Error")
   sqlContext = SQLContext(sc)

   path = 's3://2021-msia431-fei-lanqi/full_data.csv' 
   df = sqlContext.read.csv(path, header = True)

   # DATA MANIPULATION
   # feature selection
   feature_cols = ["var12","var13","var14","var15" ,"var16","var17","var18","var23",
   "var24","var25","var26","var27","var28","var34","var35","var36","var37","var38",
   "var45","var46","var47","var48","var56","var57","var58","var67","var68","var78"]
   df = df.select(["trade_id","time_stamp", "bar_num", "profit"]+feature_cols)
   
   # datetime format
   df = df.withColumn("time_stamp", to_timestamp(df.time_stamp, 'yyyy-MM-dd HH:mm:ss').alias('time_stamp')) # datetime format
   df = df.withColumn("year", year(df.time_stamp))
   df = df.withColumn("month", month(df.time_stamp))

   # datatypes casting
   for col_name in feature_cols+["bar_num","trade_id"]:
      df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
   df = df.withColumn("profit", col("profit").cast(DoubleType()))
   
   df = df.withColumn("bar_group", ((col('bar_num')-1)/10).cast(IntegerType())) # group 0: [1-10], group 1: [11-20], ...

   # feature engineering
   df2 = df
   window = Window().partitionBy(["trade_id","bar_group"]).orderBy("bar_num")
   df2 = df2.withColumn("profit_first", first(df2.profit).over(window)) # first profit over a bar group 
   df2 = df2.withColumn("profit_last", last(df2.profit).over(window)) # last profit
   df2 = df2.withColumn("profit_mean", mean(df2.profit).over(window)) # mean profit 
   df2 = df2.withColumn("profit_std", stddev(df2.profit).over(window)) # stdev of profits
   df2 = df2.filter(df2.bar_num % 10 == 0) # take one set of features for each group
   df2 = df2.withColumn("bar_group", df2.bar_group+1) # features are generated for the next bar group 
   df2 = df2.select("trade_id","bar_group","profit_first","profit_last","profit_mean","profit_std")
   df = df.join(df2, on = (df.trade_id == df2.trade_id) & (df.bar_group == df2.bar_group) , how = 'left')

   # assemble features
   feature_cols = feature_cols + ["profit_first","profit_last","profit_mean","profit_std"]
   df = df.select(feature_cols+["profit","year","month"]); df = df.na.drop(how="any")
   assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")
   df = assembler.transform(df)
   df.persist()
   
   # model training (stepwise)
   mapes = []; result = ""

   for year in range(2008,2015): # (2008,2014)
      mape1, result1 = train(df, year, True)
      mapes.append(mape1); result += result1

      mape2, result2 = train(df, year, False)
      mapes.append(mape2); result += result2

   # fisrt half of year of 2015
   mape3, result3 = train(df, 2015, True)
   mapes.append(mape3); result += result3

   df.unpersist()

   average_mape = np.mean(mapes) 
   avg_result = "Average MAPE: %.2f\n" % average_mape; result += avg_result; print(avg_result)
   min_mape = np.min(mapes)
   min_result = "Minimum MAPE: %.2f\n" % min_mape; result += min_result; print(min_result)
   max_mape = np.max(mapes)
   max_result = "Maximum MAPE: %.2f\n" % max_mape; result += max_result; print(max_result)
   
    
   # write to file
   s3 = boto3.resource('s3')
   object = s3.Object('2021-msia431-fei-lanqi', 'Exercise1.txt')
   object.put(Body=result)


    


