# spark-sentiment-analysis
Benchmarking sentiment analysis run on AWS EMR cluster.

### Dataset
Dataset is available [here](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download).

# Guide for Running a Script on Amazon EMR

## Create an S3 Bucket and Make It Public
1. Log into your AWS Management Console.
2. Navigate to the Amazon S3 service.
3. Create a new bucket and configure its permissions to allow public access. Be cautious with public settings to avoid unintended access to your data.

## Upload Files to S3 Bucket
1. Upload `bootstrapper.sh`, your main script, and dataset to the newly created S3 bucket.
2. Ensure that the uploaded files are publicly accessible if required.

## Create an EMR Cluster
1. In the AWS Management Console, go to the Amazon EMR service.
2. Start the cluster creation process.
3. Select EMR version `emr-7.8.0`.
4. Install applications: Hadoop 3.4.1, Hive 3.1.3, JupyterEnterpriseGateway 2.6.0, Livy 0.8.0, Spark 3.5.4, TensorFlow 2.16.1, and Zeppelin 0.11.1.
5. Choose the instance type as specified in Appendix E.
6. Configure the number of cores between 4 and 8.
7. Add the bootstrapper script from your S3 bucket as detailed in Appendix A.
8. Set the roles to their default settings.

## Add a Step to EMR Cluster
1. Name the step.
2. Choose the script to run from your S3 bucket.
3. Add Spark-submit options as described in Appendix B.

## Access Results
1. After the script execution, results will be stored in the specified directory of your S3 bucket.
2. Access the results by navigating to the appropriate S3 bucket directory.


# Guide for Running Locally

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




## Appendix A

```bash
sudo pip install numpy
sudo pip install pandas
sudo pip install scikit-learn
sudo pip install matplotlib
sudo pip install keras
sudo pip install statsmodels
sudo pip install nltk
sudo pip install spark-nlp
sudo pip install boto3
sudo pip install psutil
```
## Appendix B

```bash
--deploy-mode cluster --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3 
--driver-memory 8g --executor-memory 8g --executor-cores 4
```

