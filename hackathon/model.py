from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import sys

import matplotlib.pyplot as plt

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    
    for x in list_extract:
        x['score'] = featureImp[x['idx']]

    list_extract = [(x['name'],x['score']) for x in list_extract]; list_extract.sort(key = lambda x: x[1], reverse=True)
    list_extract = list_extract[:8]
    D = dict(list_extract) 

    plt.figure(figsize=(18,7))
    plt.bar(*zip(*D.items())) # when we use dict
    plt.savefig("feature_importances.png")

if __name__ == "__main__":

    sc = SparkContext(appName="abc_model"); sc.setLogLevel("Error")
    sqlContext = SQLContext(sc)

    path = sys.argv[1]
    df = sqlContext.read.csv(path, header = True)
    fleet = sqlContext.read.csv(sys.argv[2], header = True)

    # data manipulation
    df = df.join(fleet,df["truck_number_one"] == fleet.truck_number)
    df = df.withColumn("label", when(col("basic").isNull(),0).otherwise(1))
    df = df.withColumn("time_weight", df["time_weight"].cast(DoubleType()))
    df = df.withColumn("purchase_price", df["purchase_price"].cast(DoubleType()))
    df = df.withColumn("model_year", df["model_year"].cast(DoubleType()))
    df = df.select("time_weight","state","purchase_price","model_year","manufacturer","label")

    # drop nas
    # df = df.na.drop(how="any")


    # index and one-hot encoding
    stateIdxer = StringIndexer(inputCol="state", outputCol="stateIdx").fit(df)
    df = stateIdxer.transform(df)
    manuIdxer = StringIndexer(inputCol="manufacturer", outputCol="manuIdx").fit(df)
    df = manuIdxer.transform(df)

    encoder = OneHotEncoder(inputCols=["stateIdx","manuIdx"], outputCols=["stateVec","manuVec"], handleInvalid="keep").fit(df)
    df = encoder.transform(df)

    # features assembling
    assembler = VectorAssembler(inputCols = ["time_weight","purchase_price","model_year","stateVec", "manuVec"], outputCol = "features")
    df = assembler.transform(df)

    # train test split
    train, test = df.randomSplit([0.8, 0.2])

    # modeling
    # model = LogisticRegression(labelCol = "label", featuresCol = "features")
    # model.setThreshold(0.3174698622197929)
    # model = RandomForestClassifier(labelCol="label", featuresCol="features") # numTrees
    model = GBTClassifier(labelCol="label", featuresCol="features")
    

    model_fit = model.fit(train)
    train_predictions = model_fit.transform(train)
    test_predictions = model_fit.transform(test) # prediction, probability cols


    # feature importances
    feature_importances = model_fit.featureImportances # feature importances
    ExtractFeatureImp(feature_importances, train, "features")

    # accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    train_accuracy = evaluator.evaluate(train_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)
    print("Train Accuracy = %g" % (train_accuracy))
    print("Test Accuracy = %g" % (test_accuracy))


    # ##### LOGISTIC REGRESSION #####

    # # # Print the coefficients and intercept for logistic regression
    # # print("Coefficients: " + str(model_fit.coefficients))
    # # print("Intercept: " + str(model_fit.intercept))

    # # # Extract the summary from the returned LogisticRegressionModel instance trained
    # # trainingSummary = model_fit.summary

    # # # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    # # trainingSummary.roc.show()
    # # print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # # # Set the model threshold to maximize F-Measure
    # # fMeasure = trainingSummary.fMeasureByThreshold
    # # maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    # # bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    # #     .select('threshold').head()['threshold']
    # # print("Best Threshold: " + str(bestThreshold))
    # # model.setThreshold(bestThreshold)


    # # ##### FEATURE IMPORTANCE #####
    # # fit = model.fit(df)
    # # predictions = fit.transform(df) # prediction, target col
    # # predictions = predictions.select("prediction","label","state")

    # # # State
    # # state_importance = predictions.groupBy(df.state).agg((sum("prediction")/count("*")).alias('ExpectedPropViolations'), (sum("label")/count("*")).alias("PropViolations"))
    # # state_importance = state_importance.sort('ExpectedPropViolations',ascending=False).show()






   





        


