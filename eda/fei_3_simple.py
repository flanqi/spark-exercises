from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


import sys

import matplotlib.pyplot as pyplot


if __name__ == "__main__":
    sc = SparkContext(appName="exercise3"); sc.setLogLevel("Error")
    sqlContext = SQLContext(sc)

    path = sys.argv[1]
    df = sqlContext.read.csv(path, header = True)

    # data manipulation
    df = df.select('ID', 'Date', 'Beat', 'Primary Type', 'IUCR', 'Arrest')
    df = df.withColumn("Date", to_timestamp(df.Date, 'MM/dd/yyyy hh:mm:ss a').alias('Date'))
    df = df.withColumn("Year", year(df.Date))
    df = df.withColumn("WeekOfYear", weekofyear(df.Date))
    
    # generate violent and non-violent crime events for later model eval
    df.persist()
    violent = df.filter(df.Arrest=="true")
    non_violent = df.filter(df.Arrest=="false")
    df.unpersist()

    df = df.groupBy([df.Beat, df.Year, df.WeekOfYear]).agg(count("ID").alias("Crimes")).sort(["Beat","Year","WeekOfYear"], ascending=[True,False,False])
    violent = violent.groupBy([violent.Beat, violent.Year, violent.WeekOfYear]).agg(count("ID").alias("Crimes")).sort(["Beat","Year","WeekOfYear"], ascending=[True,False,False])
    non_violent = non_violent.groupBy([non_violent.Beat, non_violent.Year, non_violent.WeekOfYear]).agg(count("ID").alias("Crimes")).sort(["Beat","Year","WeekOfYear"], ascending=[True,False,False])
    
    # lag crime numbers as features
    window = Window().partitionBy("Beat").orderBy("Year","WeekOfYear")
    df = df.withColumn("CrimesLag1", lag(df.Crimes,1).over(window))
    df = df.withColumn("CrimesLag2", lag(df.Crimes,2).over(window))
    df = df.withColumn("CrimesNextWeek", lead(df.Crimes,1).over(window))
    violent = violent.withColumn("CrimesLag1", lag(violent.Crimes,1).over(window)).withColumn("CrimesLag2", lag(violent.Crimes,2).over(window)).withColumn("CrimesNextWeek", lead(violent.Crimes,1).over(window))
    non_violent = non_violent.withColumn("CrimesLag1", lag(non_violent.Crimes,1).over(window)).withColumn("CrimesLag2", lag(non_violent.Crimes,2).over(window)).withColumn("CrimesNextWeek", lead(non_violent.Crimes,1).over(window))

    # drop nas
    df = df.na.drop(how="any")
    violent = violent.na.drop(how="any"); non_violent = non_violent.na.drop(how="any")
    
    # ML pipelines
    assembler = VectorAssembler(inputCols = ["CrimesLag2","CrimesLag1"], outputCol = "features")
    lr = LinearRegression(featuresCol="features", labelCol="CrimesNextWeek")
    pipeline = Pipeline(stages = [assembler, lr])
    train, test = df.randomSplit([0.8, 0.2])

    model = pipeline.fit(train)
    predictions = model.transform(test)
    
    pred_violent = model.transform(violent)
    pred_non_violent = model.transform(non_violent)

    # report accuracy
    predictions.createOrReplaceTempView("predictions")
    accuracy = sqlContext.sql("""SELECT 100.0*SUM(ABS(prediction-CrimesNextWeek)/CrimesNextWeek)/COUNT(*) AS MAPE
                                 FROM predictions
                              """) # 30.00% MAPE
    overal_mape = accuracy.collect()[0]["MAPE"]

    pred_violent.createOrReplaceTempView("predictions_violent")
    accuracy_violent = sqlContext.sql("""SELECT 100.0*SUM(ABS(prediction-CrimesNextWeek)/CrimesNextWeek)/COUNT(*) AS MAPE
                                 FROM predictions_violent
                              """) # 65.60% MAPE
    violent_mape = accuracy_violent.collect()[0]["MAPE"]

    pred_non_violent.createOrReplaceTempView("predictions_non_violent")
    accuracy_non_violent = sqlContext.sql("""SELECT 100.0*SUM(ABS(prediction-CrimesNextWeek)/CrimesNextWeek)/COUNT(*) AS MAPE
                                 FROM predictions_non_violent
                              """) # 33.57% MAPE
    non_violent_mape = accuracy_non_violent.collect()[0]["MAPE"]

    results = 'Overall MAPE: {}\nMAPE for Violent Crimes: {}\nMAPE for non-Violent Crimes: {}'.format(overal_mape, violent_mape, non_violent_mape)
    
    # write to file
    with open('fei_3.txt', 'w') as f:
        f.write(results) 

    


