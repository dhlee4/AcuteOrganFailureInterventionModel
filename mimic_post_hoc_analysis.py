
from mimic_hp_training_scale import mimic_run_experiment

class mimic_post_hoc_analysis(mimic_run_experiment):

    def __init__(self,target_env=None,is_debug=True,cur_signature=""
                 ,target_disease=None,hyperparam_selection="CV"):
        if target_disease is None:
            raise Exception("Target disease not specified")
        mimic_run_experiment.__init__(self,target_env=target_env, is_debug=is_debug
                                      ,cur_signature=cur_signature
                                      ,target_disease=target_disease
                                      ,hyperparam_selection=hyperparam_selection)

    def evaluate_demographics(self, target_file=[]):
        cur_demo = self.add_demo()
        from pyspark.sql.functions import udf
        udf_age = udf(lambda x: x.toArray().tolist()[0])
        cur_demo = cur_demo.withColumn("AGE", udf_age("demo_feature"))
        cur_target_file = self.spark.read.parquet(self.out_file_name)

        anal_df = cur_target_file.select("ID").distinct().join(cur_demo, "ID")
        from pyspark.sql.functions import avg, stddev_samp, count
        anal_df.groupBy().agg(avg("AGE"), stddev_samp("AGE")).show()

        self.logger.info(cur_target_file.count())

        cur_death = self.get_hospital_death()
        self.logger.info(anal_df.count())
        anal_df.join(cur_death, "ID").groupBy("IS_DEAD").agg(count("*")).show()

    def get_all_feature_name(self):
        import json
        demo_features = json.load(open(self.json_demo_feature_dump_loc))
        non_demo_features = json.load(open(self.json_feature_dump_loc))
        return non_demo_features["non_demo_features"]+demo_features["demo_feature"]

    def corr_predicted_risks(self, target_file=[],top_lists=10, ascending=False):
        from pyspark.sql.functions import col
        cur_lab = self.obs_df
        #cur_lab.where("SOURCE <> 'VITAL'").show()
        cur_lab_def = self.def_df.where("SOURCE == 'LAB'")
        cur_lab_id = self.def_df.where("SOURCE == 'LAB'").select("ITEMID").rdd.flatMap(list).collect()
        cur_lab = cur_lab.where(col("ITEMID").isin(cur_lab_id))


        if target_file == []:
            for cur_of in [self.target_disch_col]:
                target_file.append(self.testing_result_dest_template.format(cur_of))

        pd_list = list()
        for cur_file in target_file:
            pd_list.append(self.raw_predicted_risks(cur_file, cur_lab, cur_lab_def,top_lists, ascending))

        return pd_list

    def raw_predicted_risks(self, target_file, cur_lab, cur_lab_def,top_lists=10, ascending=False):
        import pyspark

        # TODO Labs probably can be masked in mimic_data_Abstracter.
        from pyspark.sql.functions import col, datediff, corr, isnan, count, udf
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        # TODO need to abstract this
        udf_prob = udf(lambda x: float(x.toArray().tolist()[1]))

        self.logger.info("TARGET_FILE:{0}".format(target_file))
        try:
            te_result = self.spark.read.parquet(target_file) \
                .withColumn("Prob", udf_prob("Probability").cast("double"))
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.debug("FILE NOT EXISTS! {0}".format(target_file))
            return
        self.logger.info("CURRENT_FILE:{0}".format(target_file))
        corr_result = te_result.join(cur_lab, (cur_lab.ID == te_result.ID) & (
                    datediff(te_result.TIME_SPAN.TIME_TO, cur_lab.TIME_OBS) == 0)).groupBy("ITEMID").agg(
            corr(col("Prob").cast("double"), col("VALUE").cast("double")).alias("Pearson_Correlation"), count("*").alias("Num_OBS")).persist()

        return_df = corr_result.join(cur_lab_def, "ITEMID")\
            .where((~col("Pearson_Correlation").isNull()) & (~isnan("Pearson_Correlation")))\
            .orderBy(col("Pearson_Correlation") if ascending else col("Pearson_Correlation").desc()).limit(top_lists).toPandas()

        corr_result.unpersist()

        return return_df


    def evaluate_agg_prob(self):
        import pyspark
        from pyspark.sql.functions import col
        #terminal_outcome.show()

        from pyspark.sql.functions import udf,log,sum,exp
        from pyspark.ml.evaluation import BinaryClassificationEvaluator

        udf_prob = udf(lambda x: x.toArray().tolist()[1])
        cur_terminal_df = self.get_terminal_df()
        self.flatten_terminal_outcome()
        for cur_of in [self.target_disch_col]:
            self.logger.info( cur_of)
            try:
                cur_training_df = self.spark.read.parquet(self.training_result_dest_template.format(cur_of)).select("ID","TIME_SPAN",udf_prob("Probability").cast("double").alias("probability"),col("{0}_label".format(cur_of)).alias("label"))
                cur_testing_df = self.spark.read.parquet(self.testing_result_dest_template.format(cur_of)).select("ID","TIME_SPAN",udf_prob("Probability").cast("double").alias("probability"),col("{0}_label".format(cur_of)).alias("label"))
            except pyspark.sql.utils.AnalysisException as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.logger.info(message)
                self.logger.info("PROCESS")
                self.logger.debug( "{0} Not exists".format(cur_of))
                continue
            cur_tr_agg = cur_training_df.groupBy("ID").agg(sum(log(1.0-col("probability"))).alias("agg_prob")).select("ID",(1.0-exp("agg_prob")).alias("agg_prob").cast("double"))
            cur_te_agg = cur_testing_df.groupBy("ID").agg(sum(log(1.0-col("probability"))).alias("agg_prob")).select("ID",(1.0-exp("agg_prob")).alias("agg_prob").cast("double"))

            # TODO terminal_df is flattened terminal DX for now. Need to merge with other DF with ALI,AKI,ALF,AHF column separately.

            cur_tr_agg = cur_tr_agg.join(self.target_terminal_outcome_table,"ID")
            cur_te_agg = cur_te_agg.join(self.target_terminal_outcome_table,"ID")

            #cur_tr_agg.show()
            #cur_te_agg.show()

            from pyspark.sql.functions import count
            #cur_te_agg.select(cur_of).groupBy(cur_of).agg(count("*")).show()


            return cur_tr_agg, cur_te_agg

    def eval_feature_contribution(self, target_model, top_k=10):
        from pyspark.sql.functions import col
        def_df = self.get_def_df()
        cur_feature_list = self.get_all_feature_name()
        cur_importance = target_model.featureImportances.toArray().tolist()
        zipped_list = zip(cur_feature_list, cur_importance)
        processed_df = [{"raw_feature": cur_item[0], "IG": cur_item[1]} for cur_item in zipped_list]
        cur_feature_importance_df = self.spark.createDataFrame(processed_df)
        cur_feature_importance_num = cur_feature_importance_df.where(col("raw_feature").like("imp_N_%"))
        cur_feature_importance_cat = cur_feature_importance_df.where(col("raw_feature").like("C_%"))
        cur_feature_importance_demo = cur_feature_importance_df.where(
            ~((col("raw_feature").like("C_%")) | (col("raw_feature").like("imp_N_%"))))
        voca_dict = self.spark.read.parquet(self.voca_name).join(def_df, "ITEMID")
        from pyspark.sql.functions import concat, lit, split
        voca_dict = voca_dict.withColumn("raw_feature", concat(lit("C_"), col("idx")))

        cur_feature_importance_num = cur_feature_importance_num.withColumn("ITEMID",
                                                                           split("raw_feature", "_").getItem(2))

        from pyspark.sql.functions import udf
        udf_demo_value = udf(lambda x: x.split("||")[1] if len(x.split("||")) > 1 else "N/A")
        udf_demo_feature = udf(lambda x: x.split("||")[0])

        cur_feature_importance_num = cur_feature_importance_num.join(def_df, "ITEMID") \
            .withColumn("feature_type", lit("numeric")).withColumn("method_or_value",
                                                                   split("raw_feature", "_").getItem(3))
        cur_feature_importance_cat = cur_feature_importance_cat.join(voca_dict, "raw_feature") \
            .withColumn("feature_type", lit("categorical")).withColumn("method_or_value", col("VALUE"))
        cur_feature_importance_demo = cur_feature_importance_demo \
            .withColumn("feature_type", lit("demographics")) \
            .withColumn("LABEL", udf_demo_feature("raw_feature")) \
            .withColumn("method_or_value", udf_demo_value("raw_feature"))

        all_features = cur_feature_importance_num.select("IG", "ITEMID", "LABEL", "feature_type", "method_or_value") \
            .union(cur_feature_importance_cat.select("IG", "ITEMID", "LABEL", "feature_type", "method_or_value")) \
            .union(cur_feature_importance_demo.select("IG", lit("N/A").alias("ITEMID"), "LABEL", "feature_type",
                                                      "method_or_value"))

        return all_features.orderBy(col("IG").desc()).limit(top_k).toPandas()


if __name__ == "__main__":
    #move this to ipynb
    cur_evaluator = mimic_post_hoc_analysis(is_debug=False, cur_signature="MIMIC3_DEMO"
                                            , target_disease=["42731"])
    cur_evaluator.set_top_intv_k()


    # TODO Just export these values as pandas df
    cur_evaluator.evaluate_demographics()
    cur_evaluator.corr_predicted_risks()
    cur_evaluator.evaluate_agg_prob()
    # TODO add feature contribution describer
