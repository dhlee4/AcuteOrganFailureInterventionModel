mimic_csv_dir = "./mimic-iii-clinical-database-demo-1.4"
mimic_parquet_dir = "./mimic3_demo_dataset"

from glob import glob
target_csvs = glob("{0}/*.csv".format(mimic_csv_dir))

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

import os

if not os.path.isdir(mimic_parquet_dir):
    os.mkdir(mimic_parquet_dir)

from pyspark.sql.functions import count,col,lower

for cur_csv in target_csvs:
    target_df_name = cur_csv.replace("\\","/").split("/")[-1].split(".")[0].upper()
    csv_spark = spark.read.csv(cur_csv,header=True)
    if csv_spark.columns in ["HADM_ID"]:
        csv_spark.repartition("HADM_ID").write.save("{0}/{1}".format(mimic_parquet_dir,target_df_name))
    else:
        csv_spark.write.save("{0}/{1}".format(mimic_parquet_dir, target_df_name))

    spark.read.parquet("{0}/{1}".format(mimic_parquet_dir, target_df_name)).show()