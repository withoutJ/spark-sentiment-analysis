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

# Get model type from Spark configuration
model_type = spark.conf.get("spark.sentiment.model", "vivekn")  # Default to vivekn if not specified

MODEL_NAME = 'sentimentdl_use_twitter'

# Read the dataset
df = spark.read.csv("training.1600000.processed.noemoticon.csv", header=False)

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

# Create different pipeline stages based on selected model
if model_type == 'vivekn':
    sentiment_model = ViveknSentimentModel.pretrained() \
        .setInputCols(["document", "token"]) \
        .setOutputCol("sentiment")
    
    pipeline_stages = [
        document_assembler,
        tokenizer,
        normalizer,
        sentiment_model
    ]
else:  # sentimentdl
    sentence_embeddings = UniversalSentenceEncoder.pretrained() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence_embeddings")
    
    sentiment_model = SentimentDLModel.pretrained(name=MODEL_NAME, lang="en") \
        .setInputCols(["sentence_embeddings"]) \
        .setOutputCol("sentiment")
    
    pipeline_stages = [
        document_assembler,
        sentence_embeddings,
        sentiment_model
    ]

# Create and fit the pipeline
pipeline = Pipeline(stages=pipeline_stages)

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
    .csv(f"sentiment_results_{model_type}")

# Stop Spark session
spark.stop()