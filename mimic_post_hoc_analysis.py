
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

        '''corr_result.join(cur_lab_def, "ITEMID").where((~col("corr_val").isNull()) & (~isnan("corr_val"))).orderBy(
            col("corr_val")).show(100, truncate=False)
        corr_result.join(cur_lab_def, "ITEMID").where((~col("corr_val").isNull()) & (~isnan("corr_val"))).orderBy(
            col("corr_val").desc()).show(100, truncate=False)'''
        return_df = corr_result.join(cur_lab_def, "ITEMID")\
            .where((~col("Pearson_Correlation").isNull()) & (~isnan("Pearson_Correlation")))\
            .orderBy(col("Pearson_Correlation") if ascending else col("Pearson_Correlation").desc()).limit(top_lists).toPandas()

        corr_result.unpersist()

        return return_df

        '''self.logger.info("TE_AUC:{0}".format(
            BinaryClassificationEvaluator(rawPredictionCol="Prob", labelCol="label").evaluate(te_result)))
        self.logger.info("TE_PRC:{0}".format(
            BinaryClassificationEvaluator(rawPredictionCol="Prob", labelCol="label", metricName="areaUnderPR").evaluate(
                te_result)))'''

    def evaluate_agg_prob(self):
        import pyspark
        from pyspark.sql.functions import col
        #terminal_outcome.show()

        from pyspark.sql.functions import udf,log,sum,exp
        from pyspark.ml.evaluation import BinaryClassificationEvaluator

        udf_prob = udf(lambda x: x.toArray().tolist()[1])
        #of_list = {"AHF":["42821","42823","42831","42833","42841","42843"],"ALI":["51881","51884","51851","51853"],"AKI":["5845","5849","5848"],"ALF":["570"]}
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
            cur_te_agg.select(cur_of).groupBy(cur_of).agg(count("*")).show()

            print("TR_AUC:{0}".format(BinaryClassificationEvaluator(rawPredictionCol="agg_prob",labelCol=cur_of).evaluate(cur_tr_agg)))
            print("TE_AUC:{0}".format(BinaryClassificationEvaluator(rawPredictionCol="agg_prob",labelCol=cur_of).evaluate(cur_te_agg)))
            print("TR_PRC:{0}".format(BinaryClassificationEvaluator(rawPredictionCol="agg_prob",labelCol=cur_of,metricName="areaUnderPR").evaluate(cur_tr_agg)))
            print("TE_PRC:{0}".format(BinaryClassificationEvaluator(rawPredictionCol="agg_prob",labelCol=cur_of,metricName="areaUnderPR").evaluate(cur_te_agg)))

            return cur_tr_agg, cur_te_agg



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
