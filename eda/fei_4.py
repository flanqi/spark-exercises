from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *

import sys

import matplotlib.pyplot as plt

def plot_aux(D, path):
   plt.figure()
   plt.bar(*zip(*D.items())) 
   plt.savefig(path)

if __name__ == "__main__":
   sc = SparkContext(appName="exercise4"); sc.setLogLevel("Error")
   sqlContext = SQLContext(sc)

   path = sys.argv[1]
   df = sqlContext.read.csv(path, header = True)

   # data manipulation
   df = df.select('ID', 'Date', 'Arrest').filter(df.Arrest == "true")

   df = df.withColumn("Date", to_timestamp(df.Date, 'MM/dd/yyyy hh:mm:ss a').alias('Date'))
   df = df.withColumn("Month", month(df.Date))
   df = df.withColumn("DayOfWeek", dayofweek(df.Date))
   df = df.withColumn("Hour", hour(df.Date))
   
   
   df_monthly = df.groupBy(df.Month).agg(count("ID").alias("Crimes")).sort("Month")
   df_weekly = df.groupBy(df.DayOfWeek).agg(count("ID").alias("Crimes")).sort("DayOfWeek")
   df_hourly = df.groupBy(df.Hour).agg(count("ID").alias("Crimes")).sort("Hour")

   results_monthly = df_monthly.collect(); results_weekly = df_weekly.collect(); results_hourly = df_hourly.collect()
   results_monthly = [(row["Month"], row["Crimes"]) for row in results_monthly]
   results_weekly = [(row["DayOfWeek"], row["Crimes"]) for row in results_weekly]
   results_hourly = [(row["Hour"], row["Crimes"]) for row in results_hourly]
   
   D_monthly = dict(results_monthly); D_weekly = dict(results_weekly); D_hourly = dict(results_hourly)
    
   # make bar plots
   plot_aux(D_monthly, "fei_3_monthly.png")
   plot_aux(D_weekly, "fei_3_weekly.png")
   plot_aux(D_hourly, "fei_3_hourly.png")