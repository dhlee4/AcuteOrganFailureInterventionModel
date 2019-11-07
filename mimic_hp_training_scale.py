
try:
    import findspark
    findspark.init("/Users/daehyunlee/spark-2.3.2-bin-hadoop2.7/")
except:
    pass

import sys
import os
from annotator_gen import *
from pyspark.sql import SparkSession
import abc

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, concat, log,max
import abc



from data_preprocessor import data_preprocessor

from mimic_preprocess import mimic_preprocessor
from data_run_experiment import data_run_experiment

class mimic_run_experiment(mimic_preprocessor,data_run_experiment):
    def __init__(self,target_env=None,is_debug=True,cur_signature="",eval_metric="AUPRC",hyperparam_selection="CV"
                 ,target_disease=None):
        # class(processed previous stuffs?)
        if target_disease is None:
            raise Exception("Target disease not specified")
        mimic_preprocessor.__init__(self,target_env, is_debug,cur_signature)
        self.logger.info("preprocessor_init done")
        if (type(target_disease) == str) or (type(target_disease) == int):
            self.target_disch_icd9=[target_disease]
        else:
            self.target_disch_icd9 = target_disease
        data_run_experiment.__init__(self,eval_metric=eval_metric,hyperparam_selection=hyperparam_selection
                                     ,target_disch=self.target_disch_icd9)
        self.logger.info("run_experiment done")
        self.hyperparam_selection=hyperparam_selection
        self.cur_preprocessed = self.run_preprocessor()
        self.logger.info("PREPROCESSOR_OUT")
        self.cur_action_df = self.action_df
        self.cur_terminal_df = self.terminal_df
        self.annot_intv_dir = self.intermediate_dir+"/intervention_{0}_{1}"

        self.cur_demo_file_name = self.intermediate_dir + "/demo_processed"
        self.temp_missing_drop = self.out_file_name + "_imputed"

        if self.is_debug:
            self.cur_cv_fold = 2
        else:
            self.cur_cv_fold = 5


    def get_param_grid(self,cur_model_selection):
        from pyspark.ml.tuning import ParamGridBuilder
        if self.is_debug:
            return ParamGridBuilder() \
                .addGrid(cur_model_selection.maxDepth, [2]) \
                .addGrid(cur_model_selection.subsamplingRate, [0.3]) \
                .addGrid(cur_model_selection.maxIter, [2]) \
                .build()
        else:
            #20,0.5,10
            return ParamGridBuilder() \
                .addGrid(cur_model_selection.maxDepth, [2]) \
                .addGrid(cur_model_selection.subsamplingRate, [0.3,0.8]) \
                .addGrid(cur_model_selection.maxIter, [2]) \
                .build()


    def add_demo(self):
        import pyspark
        try:
            return self.spark.read.parquet(self.cur_demo_file_name).withColumnRenamed("HADM_ID", "ID")
        except pyspark.sql.utils.AnalysisException as ex:

            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")

            from pyspark.sql.functions import datediff,col
            from pyspark.ml.feature import OneHotEncoder, StringIndexer
            from pyspark.ml.feature import VectorAssembler
            cur_demo = self.spark.read.parquet(self.data_dir + "/ADMISSIONS").select("SUBJECT_ID", "HADM_ID", "ADMITTIME", "ADMISSION_TYPE", "ADMISSION_LOCATION", "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY")
            cur_pts = self.spark.read.parquet(self.data_dir + "/PATIENTS").select("SUBJECT_ID", "DOB", "GENDER")
            merged_demo = cur_demo.join(cur_pts,"SUBJECT_ID").drop("SUBJECT_ID")
            merged_demo = merged_demo.withColumn("AGE",datediff("ADMITTIME","DOB")/365.0).withColumn("AGE",when(col("AGE")>90,90).otherwise(col("AGE"))).drop("ADMITTIME","DOB").where("AGE > 18").fillna("N/A")

            target_col = merged_demo.columns
            target_col.remove("AGE")
            target_col.remove("HADM_ID")
            target_col.sort()
            self.logger.debug(target_col)
            vector_target = ["AGE"]
            demo_col_list = ["AGE"]
            for cat_col in target_col:
                SI_model= StringIndexer(inputCol=cat_col, outputCol="SI_{0}".format(cat_col)).fit(merged_demo)
                demo_col_list = demo_col_list+[demo_var+"||"+demo_info for demo_var, demo_info in (zip([cat_col]*len(SI_model.labels),SI_model.labels))]
                merged_demo = SI_model.transform(merged_demo)
                merged_demo = OneHotEncoder(inputCol="SI_{0}".format(cat_col),outputCol="OH_{0}".format(cat_col), dropLast=False).transform(merged_demo)
                vector_target.append("OH_{0}".format(cat_col))

            import json
            json.dump({"demo_feature":demo_col_list},open(self.json_demo_feature_dump_loc,"w"))
            sorted(vector_target)
            self.logger.debug( vector_target)
            return_df = VectorAssembler(inputCols=vector_target,outputCol="demo_feature").transform(merged_demo)
            return_df.write.save(self.cur_demo_file_name)
            return_df = self.spark.read.parquet(self.cur_demo_file_name).withColumnRenamed("HADM_ID", "ID").select("ID","demo_feature")
            return return_df



if __name__ == "__main__":
    from mimic_hp_training_scale import mimic_run_experiment

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_env", help="target environment, default=Test")
    args = parser.parse_args()
    if args.target_env:
        cur_target_env = args.target_env
    else:
        cur_target_env = None

    #Target Disease: 42731, 5849, 51881,5990
    cur_experiment = mimic_run_experiment(target_env=cur_target_env,is_debug=False, cur_signature="MIMIC3_DEMO"
                                          ,eval_metric="AUPRC",hyperparam_selection="TVT",target_disease=["51881"])
    cur_experiment.logger.debug("IN")
    for cur_intv_num in [10]:
        cur_experiment.logger.debug("run_exp:{0}".format(cur_intv_num))
        cur_experiment.run_experiment(num_intv = cur_intv_num)
    cur_experiment.logger.debug("exp_done:{0}".format(cur_intv_num))



