from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint

import sys
import re

def parse_aux(line): # string -> (year, 1)
    samples = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', line)
    return (int(samples[17]),1) 

def filter_na(line): 
    if line[0] is None:
        return False
    else:
        return True


def add_label(line): # (year, num_crimes) -> LabeledPoint(mayor,[year,num_crimes])
    if line[0] >= 2011:
        return LabeledPoint(1,[line[1]]) # Emaneul
    else:
        return LabeledPoint(0, [line[1]]) # Daly

if __name__ == "__main__":
    # create RDD from csv
    sc = SparkContext(appName="exercise2_3"); sc.setLogLevel("Error")
    sqlContext = SQLContext(sc)

    lines = sc.textFile(sys.argv[1]); lines.persist()
    header = lines.first()
    lines_filtered = lines.filter(lambda row: row != header) # discard first row
    lines.unpersist()

    data = lines_filtered.map(parse_aux).filter(lambda l: l[0]<2020) # keep only relevant columns (year, 1); Emaneul was no longer mayor in 2020
    data_filtered = data.filter(filter_na) # drop na

    crimes_by_year = data_filtered.reduceByKey(lambda a,b: a+b) # (year,num_crimes)

    crimes_by_mayor = crimes_by_year.map(add_label) # LabeledPoint(mayor,[num_crimes])

    chiSqTest = Statistics.chiSqTest(crimes_by_mayor)
    
    for i, result in enumerate(chiSqTest):
        print(result)