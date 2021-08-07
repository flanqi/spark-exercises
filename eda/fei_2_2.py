from pyspark import SparkContext
from pyspark.mllib.stat import Statistics

import sys
import re

import numpy as np

most_recent_year = None

def parse_line(line):
    samples = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', line)
    return (samples[10],int(samples[17])) # (beat, year)

def recent_5_year(line):
    notNone = line[1] is not None
    if notNone:
        if line[1] >= most_recent_year-4:
            return True

    return False

def check_none(line):
    if line[1] is None:
        return False
    else:
        return True

def max_aux(line):
    first = line[1][0]; second = line[1][1]

    if second is None: # due to left outer join
        return (line[0],first)
    else:
        return (line[0], max(first, second))


def corr_aux(line):
    series = line[1] # [(beat, num_crimes),..]
    series.sort(key=lambda elem: elem[0]) # sort list by beat
    series = [x[1] for x in series] # take num_crimes

    return series # [num1, num2, ...] ordered by beat for a given year

def write_to_string(ls):
    result = ""

    for (pair, correlation) in ls:
        result = result + pair[0]+","+pair[1] + ": " + str(correlation) + "\n"
    
    return result

if __name__ == "__main__":
    # create RDD from csv
    sc = SparkContext(appName="exercise2_2"); sc.setLogLevel("Error")
    lines = sc.textFile(sys.argv[1]); lines.persist()
    header = lines.first()
    lines_filtered = lines.filter(lambda row: row != header) # discard first row
    lines.unpersist()

    data = lines_filtered.map(parse_line) # keep only relevant columns

    # get most recent year
    maxRDD = data.map(lambda l: ("most_recent_year",l[1])).reduceByKey(lambda a,b: max(a,b))
    most_recent_year = maxRDD.collect()[0][1]

    # replace nas with 0 
    yearRDD = sc.parallelize(list(range(most_recent_year-4,most_recent_year+1))) # fix 5 year range
    beatRDD = data.map(lambda l: l[0]).distinct(); beatRDD.persist() # fix list of beats
    emptyRDD = beatRDD.cartesian(yearRDD).map(lambda l: (l,0)) # RDD[((beat, year),0)]
    beats = beatRDD.collect(); beats.sort(); beatRDD.unpersist() # store beat names for later retrieval

    crimes = data.filter(recent_5_year).map(lambda l: ((l[0],l[1]),1)) # RDD[((beat, year_with_missing),1)]
    data.unpersist()

    crimes_merged = emptyRDD.leftOuterJoin(crimes)
    crimes_imputed = crimes_merged.map(max_aux) # RDD[((beat,year), 1 or 0)] 
    
    # sum the number of crime cases per (beat,year)
    crimes_summed = crimes_imputed.reduceByKey(lambda a,b: a+b).map(lambda l: (l[0][1],(l[0][0],l[1]))) # RDD[(year, (beat, num_crimes))]
    crimes_grouped = crimes_summed.groupByKey().mapValues(list).sortBy(lambda l: l[0]) # RDD[(year, [(b1,n1),(b3,n3),(b2,n2)...])]; rows are sorted by beat

    crimes_matrix = crimes_grouped.map(corr_aux) # RDD[[n1,n2,...]]; n_i is the num_crimes for beat_i, the list is sorted by beat names; rows are sorted by year

    # calcalate the correlation matrix
    corr = Statistics.corr(crimes_matrix, method="pearson") 
    # np.savetxt('corr2.txt', corr, delimiter=',') # save matrix to txt
    dim = len(beats) # dim is the number of beats, i.e., corr is of dimension dim x dim
    results = []
    for i in range(dim):
        for j in range(i+1,dim):
            if np.isnan(corr[i][j]):
                results.append([(beats[i],beats[j]),0])
            else:
                results.append([(beats[i],beats[j]),corr[i][j]])

    results.sort(key = lambda l: abs(l[1]), reverse=True)
    with open("fei_2_2.txt", "w") as output:
        output.write(write_to_string(results))



    