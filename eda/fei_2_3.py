from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint

import sys
import re

import numpy as np

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
        return (1,[line[1]]) # Emaneul
    else:
        return (0,[line[1]]) # Daly

def t_test(rdd):
    # two sample t-test unequal variance
    results = rdd.collect()
    
    daly = results[0][1]; m1 = np.mean(daly); n1 = len(daly)
    emaneul = results[1][1]; m2 = np.mean(emaneul); n2 = len(emaneul)
    s1_2 = sum((daly-m1)*(daly-m1))/(n1-1); s2_2 = sum((emaneul-m2)*(emaneul-m2))/(n2-1)
    t = (m2-m1) / np.sqrt((s1_2/n1+s2_2/n2))

    return t


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

    crimes_by_mayor = crimes_by_year.map(add_label).reduceByKey(lambda a,b: a+b).sortBy(lambda l: l[0]) # (mayor label,[num_crimes, ...])

    print(crimes_by_mayor.collect())
    t_statistics = t_test(crimes_by_mayor)
    
    print(t_statistics)