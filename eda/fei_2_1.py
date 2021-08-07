from pyspark import SparkContext

import sys
import re

most_recent_year = None

def parse_line(line):
    samples = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', line)
    return [samples[3],int(samples[17])] # [block, year]

def recent_3_year(line):
    if line[1]>=most_recent_year-2:
        return True
    else:
        return False

def write_to_string(ls):
    result = ""

    for (block, num_crimes) in ls:
        result = result + block + "," + str(num_crimes) + "\n"
    
    return result

if __name__ == "__main__":
    # create RDD from csv
    sc = SparkContext(appName="exercise2_1"); sc.setLogLevel("Error")
    lines = sc.textFile(sys.argv[1]); lines.persist()
    header = lines.first()
    lines_filtered = lines.filter(lambda row: row != header) # discard first row
    lines.unpersist()

    data = lines_filtered.map(parse_line) # keep only relevant columns
    data.persist()
    
    # get most recent year
    maxRDD = data.map(lambda l: ("most_recent_year",l[1])).reduceByKey(lambda a,b: max(a,b))
    most_recent_year = maxRDD.collect()[0][1]

    # sum the number of crime cases per block
    crimes_by_block = data.filter(recent_3_year).map(lambda l: (l[0],1)).reduceByKey(lambda a,b: a+b).sortBy(lambda l: l[1], ascending=False)
    results = crimes_by_block.take(10) # take the first 10

    data.unpersist()

    # write to file
    with open("fei_2_1.txt", "w") as output:
        output.write(write_to_string(results))


    