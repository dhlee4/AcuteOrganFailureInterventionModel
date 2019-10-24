import sys
import os
from annotator_gen import *
from pyspark.sql import SparkSession
import abc
# TODO expose annotation criteria so that it can be changed! also run on 'Improving documentation and coding for acute
# organ dysfunction biases estimates of .... by Chanu Rhee et al., from Critical Care/ Maybe in annotator_gen.py
# TODO define quick turn hyperparameter space.
# TODO incorporate misc_anal/AHF_focused_anal.py, merge_all_pts.py into validation routine
# TODO validation routine for verifying the number of instances and number of patients matches across the models
# TODO validation routine for verifying training IDs and testing IDs are consistent across models

# TODO add output signature and temp signature for generated data in the higher level so that overwrite or collision can
# TODO be avoided afterwards

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, concat, log,max
import abc



from data_preprocessor import data_preprocessor

from mimic_preprocess import mimic_preprocessor
from data_run_experiment import data_run_experiment

class mimic_run_experiment(mimic_preprocessor,data_run_experiment):
    '''
    Evaluator will be separated!
    '''

    def __init__(self,target_env=None,is_debug=True,cur_signature=""):
        # class(processed previous stuffs?)
        mimic_preprocessor.__init__(self,target_env, is_debug,cur_signature)
        self.logger.info("preprocessor_init done")
        data_run_experiment.__init__(self)
        self.logger.info("run_experiment done")

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

        self.testing_result_dest_template = self.home_dir + "/{3}_0.7_{0}_TEST_RESULT_{1}_{2}".format("{0}", self.postfix, self.add_flag,self.cur_signature )
        self.training_result_dest_template = self.home_dir + "/{3}_0.7_{0}_TR_RESULT_{1}_{2}".format("{0}", self.postfix, self.add_flag,self.cur_signature )
        self.model_dir_template = self.home_dir + "/{4}_{0}_GB_0.7_{1}_{2}_{3}".format("{0}", self.postfix, self.add_flag, "{1}",self.cur_signature )
        self.annot_intv_dir = self.intermediate_dir+"/intervention_{0}_{1}"





    def add_demo(self):#(tr,te# ):
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
            for cat_col in target_col:
                SI_model= StringIndexer(inputCol=cat_col, outputCol="SI_{0}".format(cat_col)).fit(merged_demo)
                merged_demo = SI_model.transform(merged_demo)
                merged_demo = OneHotEncoder(inputCol="SI_{0}".format(cat_col),outputCol="OH_{0}".format(cat_col), dropLast=False).transform(merged_demo)
                vector_target.append("OH_{0}".format(cat_col))

            sorted(vector_target)
            self.logger.debug( vector_target)
            return_df = VectorAssembler(inputCols=vector_target,outputCol="demo_feature").transform(merged_demo)
            return_df.write.save(self.cur_demo_file_name)
            return_df = self.spark.read.parquet(self.cur_demo_file_name).withColumnRenamed("HADM_ID", "ID").select("ID","demo_feature")
            return return_df



if __name__ == "__main__":
    '''
    IDEAL CALL
    
    --
    cur_experiment = run_experiment(MIMIC3, TEST_ENV)
    print cur_experiment
      - Running ID
    print cur_experiment.chk_intermediary_dir()
    print cur_experiment.chk_final_dir()
    print cur_experiment.run_mimic_experiment()
    cur_model = cur_experiment.get_best_model()
    print cur_experiment.show_feature_contrib()
    print cur_experiment.show_agreement_lab_test()
    print cur_experiment.show_tr_eval(level="pts",rawMetrics="AUPRC")
    print cur_experiment.show_te_eval(level="pts",raw_metrics="AUROC")
    
    ## Further evaluation should be conducted in separate codes.
    
    --
    '''
    from mimic_hp_training_scale import mimic_run_experiment

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_env", help="target environment, default=Test")
    args = parser.parse_args()
    if args.target_env:
        cur_target_env = args.target_env
    else:
        cur_target_env = None

    cur_experiment = mimic_run_experiment(target_env=cur_target_env,is_debug=False, cur_signature="MIMIC3_DEMO")
    cur_experiment.logger.debug("IN")
    for cur_intv_num in [5,10]:
        cur_experiment.logger.debug("run_exp:{0}".format(cur_intv_num))
        cur_experiment.run_experiment(num_intv = cur_intv_num)
    cur_experiment.logger.debug("exp_done:{0}".format(cur_intv_num))

