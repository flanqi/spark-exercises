from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

import sys
import re

# import pandas as pd
import matplotlib.pyplot as plt

def parse_line(line):
    one_sample = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', line) # skip commas in quotes
    month = one_sample[2][0:2]
    one_sample.append(int(month)) # add month column
    return one_sample 

if __name__ == "__main__":
    # create RDD from csv
    sc = SparkContext(appName="exercise1"); sc.setLogLevel("Error")
    sqlContext = SQLContext(sc)

    lines = sc.textFile(sys.argv[1]); lines.persist()
    header = lines.first(); header_ls = header.split(",") # extract header
    lines_filtered = lines.filter(lambda row: row != header) # discard first row
    lines.unpersist()

    data = lines_filtered.map(parse_line) # RDD[list]
    data.persist() 

    # create schema and register dataframe as a table
    fields = [StructField(field_name, StringType(), True) for field_name in header_ls] + [StructField("Month", IntegerType(), True)]
    schema = StructType(fields)

    schemaCrime = sqlContext.createDataFrame(data, schema); data.unpersist()
    schemaCrime.registerTempTable("crime")

    # query results
    results = sqlContext.sql("SELECT Month, COUNT(DISTINCT ID) AS NumCrimes FROM crime GROUP BY Year, Month ORDER BY Month;") # dataframe
    results.createOrReplaceTempView("tmp")
    results = sqlContext.sql("SELECT Month, AVG(NumCrimes) AS AvgCrimes FROM tmp GROUP BY Month ORDER BY Month;")

    # results.write.csv(sys.argv[2]) # write to file
    results = results.collect() # list of Row
    results = [(row["Month"], row["AvgCrimes"]) for row in results]
    D = dict(results) # convert to dictionary
    
    # make bar plot
    # plt.bar(df.Month, df.NumCrimes) # when df is pandas DF
    # plt.show()
    plt.bar(*zip(*D.items())) # when we use dict
    plt.savefig("fei_1.png")
    