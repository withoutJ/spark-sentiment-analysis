# spark-sentiment-analysis
Benchmarking sentiment analysis run on AWS EMR cluster.

### Dataset
Dataset is available [here](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download).

### Requirements

```bash
pip install pyspark spark-nlp==5.5.3
```

### Inference

```bash
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 --conf spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 main.py
```
