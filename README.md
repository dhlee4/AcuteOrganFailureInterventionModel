# AcuteOrganFailureInterventionModel

To execute,

1. Install Apache Spark
2. Download MIMIC-3 demo dataset from https://physionet.org/content/mimiciii-demo/1.4/
3. Extract csv files under ./mimic-iii-clinical-database-demo-1.4
4. Run parquetizer.py with the following command:

`spark-submit parquetizer.py`

5. Run model trainer mimic_hp_training_scale.py with the following command:

`spark-submit mimic_hp_training_scale.py`
