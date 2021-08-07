from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline


import sys



if __name__ == "__main__":
   sc = SparkContext(appName="exercise3"); sc.setLogLevel("Error")
   sqlContext = SQLContext(sc)

   path = sys.argv[1]
   df = sqlContext.read.csv(path, header = True)
   external = sqlContext.read.csv(sys.argv[2], header = True) # external dataset containing community population and areas; source https://en.wikipedia.org/wiki/Community_areas_in_Chicago

   # data manipulation
   df = df.select('ID', 'Date', 'Beat', 'Community Area', 'Arrest')
   df = df.withColumn("Date", to_timestamp(df.Date, 'MM/dd/yyyy hh:mm:ss a').alias('Date'))
   df = df.withColumn("Year", year(df.Date))
   df = df.withColumn("WeekOfYear", weekofyear(df.Date))
    
   external = external.select('CommunityNo', 'Area_km2')
   external = external.withColumn("Area", external["Area_km2"].cast(DoubleType()))

   df = df.join(external, df["Community Area"] == external.CommunityNo, 'leftouter')
   df = df.groupBy([df.Beat, df.Year, df.WeekOfYear, df.Area]).agg(count("ID").alias("Crimes"), count(when(col("Arrest")=="true", True)).alias('ArrestCount')).sort(["Beat","Year","WeekOfYear"], ascending=[True,False,False])
   df = df.withColumn("Violent", when(col("ArrestCount")>0.5*col("Crimes"),"true").otherwise("false")) # determine violent / nonviolent 

   # lag crime numbers as features
   window = Window().partitionBy("Beat").orderBy("Year","WeekOfYear")
   df = df.withColumn("CrimesLag1", lag(df.Crimes,1).over(window))
   df = df.withColumn("CrimesLag2", lag(df.Crimes,2).over(window))
   df = df.withColumn("CrimesLag3", lag(df.Crimes,3).over(window))
   df = df.withColumn("CrimesLag4", lag(df.Crimes,4).over(window))
   df = df.withColumn("CrimesNextWeek", lead(df.Crimes,1).over(window))

   # index and one-hot encode beat
   beatIdxer = StringIndexer(inputCol="Beat", outputCol="BeatIdx")
   encoder = OneHotEncoder(inputCols=["BeatIdx"], outputCols=["BeatVec"], handleInvalid="keep")

   # drop nas
   df = df.na.drop(how="any")
    
   # ML pipelines
   assembler = VectorAssembler(inputCols = ["CrimesLag4","CrimesLag3","CrimesLag2","CrimesLag1","BeatVec","Area"], outputCol = "features")
   rf = RandomForestRegressor(featuresCol="features", labelCol="CrimesNextWeek")
   pipeline = Pipeline(stages = [beatIdxer, encoder, assembler, rf])
   train, test = df.randomSplit([0.8, 0.2])

   model = pipeline.fit(train)
   predictions = model.transform(test)

   # report accuracy
   predictions.createOrReplaceTempView("predictions")
   accuracy = sqlContext.sql("""SELECT 100.0*SUM(ABS(prediction-CrimesNextWeek)/CrimesNextWeek)/COUNT(*) AS MAPE
                                 FROM predictions
                             """)
   overal_mape = accuracy.collect()[0]["MAPE"]

   accuracy_violent = sqlContext.sql("""SELECT 100.0*SUM(ABS(prediction-CrimesNextWeek)/CrimesNextWeek)/COUNT(*) AS MAPE
                                        FROM predictions
                                        WHERE Violent = "true"
                                     """) 
   violent_mape = accuracy_violent.collect()[0]["MAPE"]

   accuracy_non_violent = sqlContext.sql("""SELECT 100.0*SUM(ABS(prediction-CrimesNextWeek)/CrimesNextWeek)/COUNT(*) AS MAPE
                                            FROM predictions
                                            WHERE Violent = "false"
                                         """) 
   non_violent_mape = accuracy_non_violent.collect()[0]["MAPE"]

   results = 'Overall MAPE: {}\nMAPE for Violent Crimes: {}\nMAPE for non-Violent Crimes: {}'.format(overal_mape, violent_mape, non_violent_mape)
    
   # write to file
   with open('fei_3.txt', 'w') as f:
      f.write(results) 

    


