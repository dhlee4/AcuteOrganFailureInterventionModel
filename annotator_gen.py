
from spark_and_logger import spark_and_logger

class annotator_gen():
    def __init__(self):
        cur_spark_and_logger = spark_and_logger()
        self.spark = cur_spark_and_logger.spark
        self.logger = cur_spark_and_logger.logger

    @staticmethod
    def sep_dx_code_def(x):
        ret_x = x.strip()
        return [str(ret_x.split(" ")[0]), (" ".join(ret_x.split(" ")[1:]))]
    #    return (" ".join(ret_x.split(" ")[1:])).strip()

    def identify_terminal_without_action(self,terminal_outcome, action_df,actionItems=[],actionCol="ITEMID",terminalCol="TOUTCOME",idCol="ID"):
        from pyspark.sql.functions import collect_list,col
        cur_pos_object = terminal_outcome.where("{0}==1".format(terminalCol))
        cur_censor_object = cur_pos_object.join(action_df.where(col(actionCol).isin(actionItems)),idCol,"outer").where(col(actionCol).isNull()).select(idCol).distinct()
        return cur_censor_object




    def prep_TR_TE(self,merged_df, per_instance=False, tr_prop=0.9,targetCol="ID",test_id_list = []):
        from pyspark.sql.functions import col
        if len(test_id_list) != 0:
            tr_inst = merged_df.where(~col(targetCol).isin(test_id_list))
            te_inst = merged_df.where(col(targetCol).isin(test_id_list))
            return (tr_inst,te_inst)
        if per_instance:
            tr_inst, te_inst = merged_df.randomSplit([tr_prop, 1-tr_prop])
        else:
            tr_id, te_id = merged_df.select(targetCol).distinct().randomSplit([tr_prop,1-tr_prop])
            tr_id = tr_id.rdd.flatMap(list).collect()
            te_id = te_id.rdd.flatMap(list).collect()
            tr_inst = merged_df.where(col(targetCol).isin(tr_id))
            te_inst = merged_df.where(col(targetCol).isin(te_id))
        return (tr_inst,te_inst)

    @staticmethod
    def conduct_binomial_test(x, n, p):
        from scipy.stats import binom_test
        if x < n:
            return float(binom_test(x, n, p, "greater"))
        else:
            return None

    def find_outcome_specific_action(self, action_df, terminal_outcome,idCol="ID",actionCol="ITEMID",terminalCol="TOUTCOME"):

        from pyspark.sql.types import DoubleType
        from pyspark.sql.functions import count,udf,col,max

        target_id = action_df.select(idCol).distinct().rdd.flatMap(list).collect()
        terminal_outcome = terminal_outcome.where(col(idCol).isin(target_id)).select(idCol,terminalCol).groupBy(idCol).agg(max(terminalCol).alias(terminalCol)).persist()
        all_inst_count = len(target_id)
        pop_action_prop = action_df.select(idCol,actionCol).distinct().groupBy(actionCol).agg(count("*").alias("action_cnt")).withColumn("action_prop", col("action_cnt")/all_inst_count)
        outcome_specific_action_prop = action_df.select(idCol,actionCol).distinct().join(terminal_outcome, idCol).groupBy(actionCol,terminalCol).agg(count("*").alias("outcome_action_cnt"))
        pop_outcome_prop = terminal_outcome.select(idCol,terminalCol).distinct().groupBy(terminalCol).agg(count("*").alias("outcome_cnt"))
        action_outcome_prop_joined = outcome_specific_action_prop.join(pop_action_prop,actionCol).join(pop_outcome_prop,terminalCol)
        udf_binom_test = udf(self.conduct_binomial_test,DoubleType())
        ret_data_frame = action_outcome_prop_joined.withColumn("p_val",udf_binom_test("outcome_action_cnt","outcome_cnt","action_prop"))
        ret_data_frame = ret_data_frame.withColumn("P_OF_GIVEN_INTV",col("outcome_action_cnt")/col("action_cnt"))
        return ret_data_frame.where("{0} == 1".format(terminalCol))

    def assert_one_and_map(key_item):
        return {"A_"+key_item:1.0}
    @staticmethod
    def update_dicts(a,b):
        ret_dict = dict()
        for i in a:
            ret_dict.update(i)
        for i in b:
            ret_dict.update(i)
        return ret_dict

    def assign_action_to_outcome(self,candidate_action_dict, actionCol="ITEMID",weightCol="p_val",weightType="p_val"):
        import pandas as pd
        cur_list = list()
        for keys in candidate_action_dict.keys():
            for cur_items in candidate_action_dict[keys]:
                cur_list.append({"OUTCOME":keys, "ACTION":cur_items[actionCol], "WEIGHT":cur_items[weightCol]})
        cur_df = pd.DataFrame(cur_list)
        self.logger.debug (cur_df)
        annotation_label = cur_df.loc[cur_df.groupby("ACTION")["WEIGHT"].idxmin()]
        self.logger.debug (annotation_label)
        annotation_target = annotation_label.groupby('OUTCOME')['ACTION'].apply(list)
        annotation_target.columns=['OUTCOME','ACTION_LIST']
        self.logger.debug(annotation_target.to_json(orient='index'))
        return annotation_target.to_json(orient='index')

    def get_annotation(self,action_df,terminal_outcome,idCol="ID",terminalCol="TOUTCOME", actionCol="ITEMID", actionItems=["1","2","3","4"]):
        from pyspark.sql.functions import col
        target_action_df = action_df.where(col(actionCol).isin(actionItems))
        target_terminal_outcome = terminal_outcome.where("{0} == 1".format(terminalCol))
        ret_df = target_terminal_outcome.join(target_action_df,idCol)
        return ret_df

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    import json

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    home_dir = "/Users/dhlee4/mimic3_data/"#/test_INPUTEVENTS_MV"

    def_item = spark.read.parquet(home_dir+"D_ITEMS")

    cur_input_item = spark.read.parquet(home_dir+"test_INPUTEVENTS_MV").select(col("HADM_ID").alias("ID"),"ITEMID",col("STARTTIME").cast("timestamp").alias("TIMESTAMP"))
    cur_proc_item = spark.read.parquet(home_dir+"test_PROCEDUREEVENTS_MV").select(col("HADM_ID").alias("ID"),"ITEMID",col("STARTTIME").cast("timestamp").alias("TIMESTAMP"))
    action_original_df = cur_input_item.unionAll(cur_proc_item).select("ID","ITEMID","TIMESTAMP")
    action_df = cur_input_item.unionAll(cur_proc_item).select("ID","ITEMID").distinct()
    action_df.show()
