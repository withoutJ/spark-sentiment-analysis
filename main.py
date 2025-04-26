import json
import pandas as pd
import numpy as np

import sparknlp
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from pyspark.sql.functions import col, explode, split, regexp_replace, lower, trim
import os

# Initialize Spark Session with Spark NLP
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
    .getOrCreate()

print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

MODEL_NAME='classifierdl_use_emotion'

# Read the dataset
df = spark.read.csv("s3://di1naza/training.1600000.processed.noemoticon.csv", header=False)

# Clean the data
df = df.select(
    df._c5.alias("text"),
    (df._c0 / 4).cast("integer").alias("label")  # Convert sentiment from 0/4 to 0/1
)

# Create pipeline stages
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

vivekn_sentiment = ViveknSentimentModel.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("sentiment")

# Create and fit the pipeline
pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    vivekn_sentiment
])

# Fit the pipeline and transform the data
model = pipeline.fit(df)
result = model.transform(df)

# Show results
result.select("text", "sentiment.result").show(10, truncate=False)

# Convert array to string and save results
result.select("text", F.array_join("sentiment.result", ",").alias("sentiment")) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("s3://di1naza/sentiment_results")

# Stop Spark session
spark.stop()
