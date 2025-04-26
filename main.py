import json
import pandas as pd
import numpy as np
import time
import psutil
import os
from datetime import datetime

import sparknlp
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from pyspark.sql.functions import col, explode, split, regexp_replace, lower, trim
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import boto3
import io

# Model selection
MODEL_CHOICE = "sentimentdl"  # Options: "vivekn" or "sentimentdl"

# Initialize Spark Session with Spark NLP and monitoring
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
    .config("spark.metrics.conf.*.sink.prometheusServlet.class", "org.apache.spark.metrics.sink.PrometheusServlet") \
    .config("spark.metrics.conf.*.sink.prometheusServlet.path", "/metrics/prometheus") \
    .config("spark.metrics.conf.*.sink.prometheusServlet.port", "4040") \
    .config("spark.metrics.namespace", "sentiment_analysis") \
    .getOrCreate()

# Initialize metrics dictionary
metrics = {
    "start_time": time.time(),
    "memory_usage": [],
    "cpu_usage": [],
    "stage_times": {},
    "model_metrics": {}
}

def log_metrics(stage_name):
    """Log resource usage metrics for a specific stage"""
    metrics["memory_usage"].append({
        "stage": stage_name,
        "memory_used": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
        "timestamp": time.time()
    })
    metrics["cpu_usage"].append({
        "stage": stage_name,
        "cpu_percent": psutil.cpu_percent(),
        "timestamp": time.time()
    })

def log_stage_time(stage_name, start_time):
    """Log execution time for a specific stage"""
    end_time = time.time()
    metrics["stage_times"][stage_name] = end_time - start_time

print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

# Start data loading stage
data_load_start = time.time()
log_metrics("data_loading_start")

# Read the dataset
df = spark.read.csv("training.1600000.processed.noemoticon.csv", header=False)

# Clean the data
df = df.select(
    df._c5.alias("text"),
    df._c0.alias("raw_label")  # Keep original label for filtering
)

# Add text preprocessing
df = df.withColumn("text", F.lower(F.col("text")))  # Convert to lowercase
df = df.withColumn("text", F.regexp_replace(F.col("text"), r"http\S+|www\S+|https\S+", ""))  # Remove URLs
df = df.withColumn("text", F.regexp_replace(F.col("text"), r"@\w+", ""))  # Remove mentions
df = df.withColumn("text", F.regexp_replace(F.col("text"), r"#\w+", ""))  # Remove hashtags
df = df.withColumn("text", F.regexp_replace(F.col("text"), r"RT\s+", ""))  # Remove RT
df = df.withColumn("text", F.regexp_replace(F.col("text"), r"[^\w\s]", ""))  # Remove special characters
df = df.withColumn("text", F.regexp_replace(F.col("text"), r"\s+", " "))  # Remove extra whitespace
df = df.withColumn("text", F.trim(F.col("text")))  # Trim leading/trailing whitespace

# Show initial label distribution
print("\nInitial label distribution:")
df.groupBy("raw_label").count().orderBy("raw_label").show()

# Filter out neutral sentiment (label 2) and convert remaining labels
df = df.filter(df.raw_label != 2) \
    .withColumn("label", 
        F.when(df.raw_label < 2, 0)  # Convert 0 and 1 to 0 (negative)
        .when(df.raw_label > 2, 1)   # Convert 3 and 4 to 1 (positive)
        .otherwise(2)                # This should never happen due to filter
    ) \
    .drop("raw_label")  # Remove the temporary column

# Show filtered label distribution
print("\nFiltered label distribution:")
df.groupBy("label").count().orderBy("label").show()

log_stage_time("data_loading", data_load_start)
log_metrics("data_loading_end")

# Start pipeline creation stage
pipeline_start = time.time()
log_metrics("pipeline_creation_start")

# Create pipeline stages
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Choose between models
if MODEL_CHOICE == "vivekn":
    # Use ViveknSentimentModel
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized")

    sentiment_model = ViveknSentimentModel.pretrained() \
        .setInputCols(["document", "token"]) \
        .setOutputCol("sentiment")
    
    pipeline_stages = [
        document_assembler,
        tokenizer,
        normalizer,
        sentiment_model
    ]
    print("Using ViveknSentimentModel")
else:
    # Use SentimentDLModel with UniversalSentenceEncoder
    use = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentence_embeddings")

    sentiment_model = SentimentDLModel.pretrained('sentimentdl_use_twitter') \
        .setInputCols(["sentence_embeddings"]) \
        .setOutputCol("sentiment")
    
    pipeline_stages = [
        document_assembler,
        use,
        sentiment_model
    ]
    print("Using SentimentDLModel with UniversalSentenceEncoder")

# Create the pipeline
pipeline = Pipeline(stages=pipeline_stages)

log_stage_time("pipeline_creation", pipeline_start)
log_metrics("pipeline_creation_end")

# Start inference stage
inference_start = time.time()
log_metrics("inference_start")

# Transform the data using the pretrained model
result = pipeline.fit(df).transform(df)

# Extract sentiment result and convert to numeric
result = result.withColumn("sentiment_result", F.explode(F.col("sentiment")))
result = result.withColumn("sentiment_value", 
    F.when(F.col("sentiment_result.result") == "positive", 1.0)
    .otherwise(0.0))

log_stage_time("inference", inference_start)
log_metrics("inference_end")

# Start evaluation stage
eval_start = time.time()
log_metrics("evaluation_start")

# Calculate model metrics
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="sentiment_value",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(result)
metrics["model_metrics"]["accuracy"] = accuracy

# Calculate confusion matrix
confusion_matrix = result.groupBy("label", "sentiment_value").count()
metrics["model_metrics"]["confusion_matrix"] = confusion_matrix.collect()

log_stage_time("evaluation", eval_start)
log_metrics("evaluation_end")

# Start results saving stage
save_start = time.time()
log_metrics("results_saving_start")

# Show results
result.select("text", "sentiment_value").show(10, truncate=False)

# Save results
result.select("text", "sentiment_value") \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("sentiment_results")

log_stage_time("results_saving", save_start)
log_metrics("results_saving_end")

# Calculate total execution time
metrics["total_execution_time"] = time.time() - metrics["start_time"]

# Save metrics to S3
metrics_file = f"benchmark_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
results = [
    "=== Execution Summary ===",
    f"Total Execution Time: {metrics['total_execution_time']:.2f} seconds",
    f"Model Accuracy: {metrics['model_metrics']['accuracy']:.4f}",
    
    "\n=== Stage Times ==="
]
for stage, time_taken in metrics["stage_times"].items():
    results.append(f"{stage}: {time_taken:.2f} seconds")

results.extend([
    "\n=== Memory Usage ===",
    "Stage | Memory Used (MB) | Timestamp"
])
for usage in metrics["memory_usage"]:
    results.append(f"{usage['stage']} | {usage['memory_used']:.2f} | {usage['timestamp']}")

results.extend([
    "\n=== CPU Usage ===",
    "Stage | CPU Percent | Timestamp"
])
for usage in metrics["cpu_usage"]:
    results.append(f"{usage['stage']} | {usage['cpu_percent']:.1f}% | {usage['timestamp']}")

results.extend([
    "\n=== Confusion Matrix ===",
    "Actual | Predicted | Count"
])
for row in metrics["model_metrics"]["confusion_matrix"]:
    results.append(f"{row[0]} | {row[1]} | {row[2]}")

# Save detailed results
textual_output = spark.createDataFrame(results, 'string')
textual_output = textual_output.repartition(1)
textual_output.write.text(f"s3://di1naza/metrics/{metrics_file}")

# Stop Spark session
spark.stop()