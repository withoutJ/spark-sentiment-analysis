# spark-sentiment-analysis
Benchmarking sentiment analysis run on AWS EMR cluster.

### Dataset
Dataset is available [here](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download).

## Run Locally

### Requirements

Install the required python packages by running the following command. You can create python virtual environment or conda environment.

```bash
pip install -r requirements.txt
```

You can sepcify model choice ("vivekn" or "sentimentdl") in line 23 of main.py.

### Inference

Run the script by executing the following line.
```bash
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 --conf spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 main.py
```
