from data_preprocessor import data_preprocessor
from mimic_preprocess import mimic_preprocessor

# TODO Make sure DX diagnosis not included in terminal outcome
# TODO Make sure handle exceptions if it is not identified from the dataset(terminal DX)

import abc


class data_run_experiment():
    __metaclass__ = abc.ABCMeta
    def __init__(self,eval_metric,hyperparam_selection,target_disch):
        # TODO Figure out the way to push *.show() in logging.info
        from spark_and_logger import spark_and_logger
        cur_spark_and_logger = spark_and_logger()
        self.spark = cur_spark_and_logger.spark
        self.logger = cur_spark_and_logger.logger
        self.itemid = "ITEMID"
        self.sel_top=5 #as default
        self.postfix = "p_val_top_{0}".format(self.sel_top)
        self.add_flag="INTV_TOP_AUPRC_{0}".format(self.sel_top)
        self.target_disch_col = "DISCH_{0}".format("_".join(target_disch))
        self.non_feature_column = ["ID", "TIME_SPAN"]
        self.eval_performance_criteria=eval_metric
        self.eval_cv_or_tvt = hyperparam_selection
        from annotator_gen import annotator_gen
        self.cur_annotator = annotator_gen()
        self.json_feature_dump_loc = self.intermediate_dir+"/non_demo_feature_list.json"
        self.json_demo_feature_dump_loc= self.intermediate_dir + "/demo_feature_list.json"
        return

    def handle_missing(self,non_feature_col = ["ID","TIME_SPAN"]):
        #Take an hour and half for the small test set. Might be some way to improve it?
        # Anyway, it is one-time run so don't need to worry that much for now.
        import pyspark
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        try:
            ret_data_frame = self.spark.read.parquet(self.temp_missing_drop)
            self.logger.info(self.temp_missing_drop)
            return ret_data_frame
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")

            #impute only. aggregation will be done after adding demographics
            cur_df = self.spark.read.parquet(self.out_file_name)
            cur_cols = cur_df.columns
            categorical_cols = list()
            numerical_cols = list()
            for i in non_feature_col:
                cur_cols.remove(i)
            for i in cur_cols:
                if i.find("C_") == 0:
                    categorical_cols.append(i)
                else:
                    numerical_cols.append(i)

            cur_df = cur_df.fillna(0,subset = categorical_cols).repartition(400).checkpoint()
            self.logger.info(cur_df.count())

            from pyspark.ml.feature import Imputer
            imputedCols = ["imp_{0}".format(x) for x in numerical_cols]
            imputer = Imputer(inputCols = numerical_cols,outputCols = imputedCols).setStrategy("mean")
            imputer_model = imputer.fit(cur_df)
            ret_data_frame = imputer_model.transform(cur_df)
            ret_data_frame.select(non_feature_col+imputedCols+categorical_cols).show()
            ret_data_frame.select(non_feature_col+imputedCols+categorical_cols).write.save(self.temp_missing_drop)
            ret_data_frame = self.spark.read.parquet(self.temp_missing_drop)
            return ret_data_frame

    def set_top_intv_k(self,cur_top_k = 5):
        self.sel_top = cur_top_k
        self.postfix = "p_val_top_{0}".format(cur_top_k)
        self.add_flag = "INTV_TOP_AUPRC_{0}".format(cur_top_k)

        self.testing_result_dest_template = self.home_dir + "/{3}_0.7_{0}_TEST_RESULT_{1}_{2}".format("{0}",
                                                                                                      self.postfix,
                                                                                                      self.add_flag,
                                                                                                      self.cur_signature)
        self.training_result_dest_template = self.home_dir + "/{3}_0.7_{0}_TR_RESULT_{1}_{2}".format("{0}",
                                                                                                     self.postfix,
                                                                                                     self.add_flag,
                                                                                                     self.cur_signature)
        self.model_dir_template = self.home_dir + "/{4}_{0}_GB_{5}_0.7_{1}_{2}_{3}".format("{0}", self.postfix,
                                                                                           self.add_flag, "{1}",
                                                                                           self.cur_signature,
                                                                                           self.hyperparam_selection)
        self.annot_intv_dir = self.intermediate_dir + "/intervention_{0}_{1}"

        self.training_temp_dir = self.intermediate_dir + "/TRAINING_{0}_{1}_{2}".format(self.postfix, self.add_flag,self.target_disch_col)
        self.testing_temp_dir = self.intermediate_dir + "/TESTING_{0}_{1}_{2}".format(self.postfix, self.add_flag,self.target_disch_col)

        return

    @abc.abstractmethod
    def add_demo(self):
        pass

    def annotate_dataset(self,cur_df,annotation_method = "p_val"):
        '''

        :param cur_df:
        :param annotation_method:
        :param postfix:
        :param cur_top:
        :param non_feature_column:
        :param add_flag:
        :return:
        '''
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        if annotation_method == "p_val":
            return self.annotate_pval_dataset(cur_df)
        else:
            raise Exception("annotation method not implemented")

    def annotate_pval_dataset(self,cur_df):
        # Diagnosis is mixing up in mimic. Need to fix
        #annotation should be part of this process?or invoke?
        #dedicate to other process, and run new process if it is not exists, or return existing dataframe if it is already done, the way to renew them is delete and rerun it.
        #assume feature aggregation is done. Just merge all features except non-feature columns
        import pyspark
        try:
            tr_inst = self.spark.read.parquet(self.training_temp_dir)
            te_inst = self.spark.read.parquet(self.testing_temp_dir)
            return tr_inst,te_inst
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
            self.logger.debug("NOTEXISTS ANNOTATE_FILE")
            self.logger.debug("RUN_PROCESS")
        except:
            self.logger.info("TEST_PURPOSE")

        from pyspark.ml.feature import VectorAssembler
        postfix = self.postfix.format(self.sel_top)
        obs_df = cur_df

        cur_cols = obs_df.columns
        for i in self.non_feature_column:
            cur_cols.remove(i)
            self.logger.debug("feature_columns")
        cur_cols = sorted(cur_cols)
        self.logger.debug(cur_cols)
        import json

        json.dump({"non_demo_features":cur_cols},open(self.json_feature_dump_loc,"w"))


        obs_df = VectorAssembler(inputCols=cur_cols, outputCol="features_imputed").transform(obs_df)

        cur_time_list = obs_df.select("ID","TIME_SPAN")
        of_annotated = obs_df
        of_excl_training = dict()

        demo_feature = self.add_demo()

        of_annotated = VectorAssembler(inputCols=["features_imputed","demo_feature"],outputCol="features").transform(of_annotated.join(demo_feature,"ID"))

        of_annotated.show()

        from pyspark.sql.functions import col,lit,when
        self.logger.debug("ANNOTATED")

        cur_test_ids = self.get_target_test_id()
        self.logger.debug(cur_test_ids)
        # TODO CHECK why I put 'why 0' comment over here?
        self.logger.debug(len(cur_test_ids))
        tr_inst,te_inst = self.cur_annotator.prep_TR_TE(of_annotated,test_id_list = cur_test_ids)

        self.logger.debug("IDS")
        self.logger.debug(tr_inst.select("ID").distinct().count(), te_inst.select("ID").distinct().count())

        self.logger.debug("TR_TE_CNT:{0}_{1}".format(tr_inst.count(),te_inst.count()))

        train_data_ID = tr_inst.select("ID").distinct().rdd.flatMap(list).collect()

        testing_data_ID = te_inst.select("ID").distinct().rdd.flatMap(list).collect()

        self.action_df.show()

        train_action_df = self.action_df.where(col("ID").isin(train_data_ID)).persist()

        self.logger.debug(train_action_df.select("ID").distinct().count())

        train_terminal_outcome = self.terminal_outcome.where(col("ID").isin(train_data_ID)).persist()

        self.logger.debug(train_terminal_outcome.select("ID").distinct().count())

        intv_w_p_val = self.identify_relevant_action(train_action_df, train_terminal_outcome, tr_inst.select("ID").distinct().count())
        # TODO Check whether discharge DX is seep into this. Found one case.
        intv_w_p_val.join(self.def_df.where(col("SOURCE").isin(["CPT","MED","PROC"])), self.itemid).orderBy("p_val").show(100, truncate=False)

        from pyspark.sql.functions import sum,rand,max,lit
        from pyspark.ml.feature import VectorAssembler
        cur_annot_topk = self.sel_top

        self.action_df.show()
        self.terminal_outcome.show()

        annot_df = self.action_df.join(self.terminal_outcome,"ID").persist()
        annot_df.show()
        pos_inst_dict = dict()
        from pyspark.sql.functions import count
        for cur_of in [self.target_disch_col]:
            # For debug purpose, pass if target_of is not identified
            self.logger.debug(cur_of)
            intv_w_p_val.where("DISCH_DX == '{0}'".format(cur_of)).orderBy(col("p_val").cast("double")).show(50,truncate=False)
            target_annot_criteria = intv_w_p_val.where("DISCH_DX == '{0}'".format(cur_of)).orderBy(col("p_val").cast("double")).limit(cur_annot_topk)
            target_annot_criteria.write.save(self.annot_intv_dir.format(cur_of,cur_annot_topk),mode="overwrite")
            target_annot_criteria = target_annot_criteria.select(self.itemid).rdd.flatMap(list).collect()
            if len(target_annot_criteria) == 0:
                self.logger.info("NO TERMINAL DX {0} idenfieid from pts".format(cur_of))
                pos_inst_dict[cur_of] = None
                continue
            self.logger.debug(target_annot_criteria)
            self.logger.debug(len(target_annot_criteria))
            self.logger.debug("selected intv!!")
            self.def_df.where(col(self.itemid).isin(target_annot_criteria)).show(cur_annot_topk,truncate=False)
            pos_inst_dict[cur_of] = annot_df.where((col(self.itemid).isin(target_annot_criteria)) & (col("DISCH_DX") == cur_of))\
                .select("ID", col("TIME_OBS").cast("date").alias("TIME_OBS"), lit("1").cast("double").alias("{0}_label".format(cur_of)))\
                .distinct().persist()
            pos_inst_dict[cur_of].groupBy("{0}_label".format(cur_of)).agg(count("*")).show()
            from pyspark.sql.functions import broadcast

            true_inst = annot_df.where((col(self.itemid).isin(target_annot_criteria)) & (col("DISCH_DX") == cur_of))
            excl_id = annot_df.withColumn("IS_TARGET_OF",when(col("DISCH_DX") ==cur_of,lit("1").cast("double")).otherwise(lit("0").cast("double")))\
                .withColumn("IS_REL_INTV", when(col(self.itemid).isin(target_annot_criteria), lit("1").cast("double")).otherwise(lit("0").cast("double")))\
                .groupBy("ID").agg(sum("IS_TARGET_OF").alias("SUM_IS_TARGET_OF"),sum("IS_REL_INTV").alias("SUM_IS_REL_INTV"))\
                .where("(SUM_IS_TARGET_OF <> 0) AND (SUM_IS_REL_INTV == 0)").select("ID").distinct().rdd.flatMap(list).collect()
            self.logger.debug( "NUM_PTS_EXCLUDED:{0}".format(len(excl_id)))
            self.logger.debug( "TRAINING_INST_COUNT:{0}".format(tr_inst.count()))
            tr_inst = tr_inst.withColumn("TIME_OBS",col("TIME_SPAN.TIME_TO").cast("date"))\
                .withColumn("{0}_excl".format(cur_of), col("ID").isin(excl_id).cast("double")).repartition("ID","TIME_OBS")\
                .join(broadcast(pos_inst_dict[cur_of]),["ID","TIME_OBS"],"left_outer").fillna(value=0.0,subset=["{0}_label".format(cur_of)]).persist()
            print(tr_inst.count())
            #inst level
            tr_inst.groupBy("{0}_label".format(cur_of),"{0}_excl".format(cur_of)).agg(count("*")).show()
            te_inst = te_inst.withColumn("TIME_OBS",col("TIME_SPAN.TIME_TO").cast("date"))\
                .withColumn("{0}_excl".format(cur_of), col("ID").isin(excl_id).cast("double")).repartition("ID","TIME_OBS")\
                .join(broadcast(pos_inst_dict[cur_of]),["ID","TIME_OBS"],"left_outer").fillna(value=0.0, subset=["{0}_label".format(cur_of)]).persist()
            print(te_inst.count())
            te_inst.groupBy("{0}_label".format(cur_of),"{0}_excl".format(cur_of)).agg(count("*")).show()

            #pts_level
            tr_inst.groupBy("ID").agg(max("{0}_label".format(cur_of)).alias("{0}_label".format(cur_of)),max("{0}_excl".format(cur_of)).alias("{0}_excl".format(cur_of))).groupBy("{0}_label".format(cur_of),"{0}_excl".format(cur_of)).agg(count("*")).show()
            te_inst.groupBy("ID").agg(max("{0}_label".format(cur_of)).alias("{0}_label".format(cur_of)),max("{0}_excl".format(cur_of)).alias("{0}_excl".format(cur_of))).groupBy("{0}_label".format(cur_of),"{0}_excl".format(cur_of)).agg(count("*")).show()

        tr_inst.write.save(self.training_temp_dir, mode="overwrite")
        te_inst.write.save(self.testing_temp_dir, mode="overwrite")

        tr_inst = self.spark.read.parquet(self.training_temp_dir)
        te_inst = self.spark.read.parquet(self.testing_temp_dir)
        #te_inst.show()

        return (tr_inst,te_inst)

    def get_anno_preprocessed_data(self):
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        try:
            tr_data = self.spark.read.parquet(self.training_temp_dir)
            te_data = self.spark.read.parquet(self.testing_temp_dir)
            return (tr_data, te_data)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            raise Exception("NO PREPROCESSED DATASET!")

    def identify_relevant_action(self, action_df, terminal_df,cnt_pop):
        from pyspark.sql.functions import col
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        spark = self.spark
        def ret_p_val(obs,cnt_pop,p):
            from scipy.stats import binom_test
            return float(binom_test(obs,cnt_pop,p,alternative='greater'))

        from pyspark.sql.functions import count, udf,col
        from pyspark.sql.types import DoubleType

        cur_dx_cnt = self.terminal_outcome.groupBy("DISCH_DX").agg(count("*").alias("DISCH_DX_CNT")).cache()
        distinct_action_df = action_df.select("ID","ITEMID").distinct().cache()
        agg_action_df = distinct_action_df.groupBy("ITEMID").agg(count("*").alias("action_cnt")).withColumn("action_prop",col("action_cnt")/float(cnt_pop))
        agg_action_df.show()
        merged_terminal_action = terminal_df.join(distinct_action_df,"ID").groupBy("DISCH_DX","ITEMID").agg(count("*").alias("DISCH_DX_ACTION_CNT")).join(cur_dx_cnt,"DISCH_DX").join(agg_action_df,"ITEMID").persist()

        udf_binom_test = udf(ret_p_val,DoubleType())
        cur_test = merged_terminal_action.withColumn("p_val",udf_binom_test("DISCH_DX_ACTION_CNT","DISCH_DX_CNT","action_prop")).cache()
        return cur_test

    def run_RF(self,tr_inst,te_inst,model_of = []):
        from pyspark.sql.functions import col
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")


        if model_of == []:
            model_of = self.target_disch_col
        if type(model_of) == str:
            model_of = [model_of]

        self.logger.info("TARGET_OF:")
        self.logger.info(model_of)
        
        from pyspark.ml.classification import GBTClassifier as cur_model_selection

        cur_classifier = cur_model_selection(featuresCol="features",checkpointInterval=5)

        from pyspark.ml import Pipeline
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

        if self.eval_performance_criteria == "AUPRC":
            target_metric="areaUnderPR"
        elif self.eval_performance_criteria == "AUROC":
            target_metric="areaUnderROC"
        else:
            raise Exception("eval_metric should be either 'AUPRC' or 'AUROC'")


        evaluator = BinaryClassificationEvaluator(metricName=target_metric)
        paramGrid = self.get_param_grid(cur_model_selection)

        if self.eval_cv_or_tvt == "CV":
            pipeline = Pipeline(stages=[cur_classifier])
            orig_tr_inst = tr_inst
            orig_te_inst = te_inst
            self.logger.info("ORIGINAL_INSTANCES")
            # of pop_overview
            from pyspark.sql.functions import count, datediff
            from pyspark.sql.functions import udf, log, sum, exp, max
            udf_prob = udf(lambda x: x.toArray().tolist()[1])
            from pyspark.sql.functions import corr, udf, isnan
            for cur_of in model_of:
                self.logger.debug(cur_of)
                #should move this to back
                te_inst = orig_te_inst.withColumn("label", col("{0}_label".format(cur_of)).cast("double"))

                self.logger.info("TE_POP")
                te_inst.groupBy("label").agg(count("*")).show()

                tr_inst.printSchema()
                tr_val_pts_dict = self.get_target_tr_val_id()
                orig_tr_inst = tr_inst

                tr_pts = tr_val_pts_dict["TR"]
                val_pts = tr_val_pts_dict["VAL"]
                self.logger.info(tr_pts)
                self.logger.info(val_pts)

                all_training_ids = tr_pts+val_pts
                from random import shuffle
                shuffle(all_training_ids)
                import numpy as np
                print(len(all_training_ids))
                cv_id_list_full = np.array(all_training_ids)
                perform_dict = dict()
                for cur_cv_stage in range(self.cur_cv_fold):
                    tr_pts = cv_id_list_full[np.linspace(0,cv_id_list_full.shape[0]-1,cv_id_list_full.shape[0])%self.cur_cv_fold != cur_cv_stage].tolist()
                    val_pts = cv_id_list_full[np.linspace(0,cv_id_list_full.shape[0]-1,cv_id_list_full.shape[0])%self.cur_cv_fold == cur_cv_stage].tolist()
                    print(np.linspace(0,cv_id_list_full.shape[0]-1,cv_id_list_full.shape[0])%self.cur_cv_fold == cur_cv_stage)
                    print(cv_id_list_full[np.linspace(0,cv_id_list_full.shape[0]-1,cv_id_list_full.shape[0])%self.cur_cv_fold == cur_cv_stage])
                    print("VAL_ROUND_{0}_TARGET IDS:{1}".format(cur_cv_stage,val_pts))

                    tr_inst = orig_tr_inst.where(col("ID").isin(tr_pts))#.persist()
                    val_inst = orig_tr_inst.where(col("ID").isin(val_pts))#.persist()


                    self.logger.info("Excluded instances for training:{0}".format(
                        tr_inst.where(col("{0}_excl".format(cur_of)) == 1).count()))

                    tr_inst = tr_inst.where(col("{0}_excl".format(cur_of)) == 0).withColumn("label", col(
                        "{0}_label".format(cur_of)).cast("double"))
                    val_inst = val_inst.withColumn("label", col(
                        "{0}_label".format(cur_of)).cast("double"))

                    self.logger.info("TR_POP")
                    tr_inst.groupBy("label").agg(count("*")).show()

                    # NEED to change with other name, for example pipeline model.
                    pipeline_models = pipeline.fit(tr_inst, params=paramGrid)

                    for cur_model in pipeline_models:
                        val_pred = cur_model.transform(val_inst)
                        agg_prob_val = val_pred.groupBy("ID").agg(max("label").alias("label"),
                                                                  sum(log(1.0 - udf_prob("Probability"))).alias(
                                                                      "inverse_log_sum")) \
                            .select("label", (1.0 - exp(col("inverse_log_sum"))).alias("rawPrediction"))
                        agg_prob_val.show(300, truncate=False)
                        cur_pr = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label",
                                                               metricName=target_metric).evaluate(agg_prob_val)
                        if str(cur_model.stages[-1].extractParamMap()) not in perform_dict:
                            perform_dict[str(cur_model.stages[-1].extractParamMap())] = dict()
                            perform_dict[str(cur_model.stages[-1].extractParamMap())]["PERF"]=list()
                            perform_dict[str(cur_model.stages[-1].extractParamMap())]["PARAM"] = cur_model.stages[-1].extractParamMap()

                        perform_dict[str(cur_model.stages[-1].extractParamMap())]["PERF"].append(cur_pr)

                best_pf_measure = -1
                best_pf_param = None

                for key in perform_dict:
                    import numpy as np
                    test_array = np.array(perform_dict[key]["PERF"])
                    print(key, test_array.mean(), test_array.std())
                    if best_pf_measure < test_array.mean():
                        best_pf_measure = test_array.mean()
                        best_pf_param = perform_dict[key]["PARAM"]


                print("retrain model based on best hp from CV")
                print("PERF:{0}".format(best_pf_measure))
                print("HP:{0}".format(best_pf_param))
                tr_inst = orig_tr_inst.where(col("{0}_excl".format(cur_of)) == 0).withColumn("label", col(
                    "{0}_label".format(cur_of)).cast("double"))

                bestModel = pipeline.fit(tr_inst.where(col("ID").isin(cv_id_list_full.tolist())),params=[best_pf_param])[0]

                bestModel.save(self.model_dir_template.format(cur_of, best_pf_measure))

                prediction = bestModel.transform(te_inst)
                prediction.show()
                prediction.write.save(self.testing_result_dest_template.format(cur_of), mode="overwrite")
                tr_result = bestModel.transform(tr_inst).withColumn("Prob",udf_prob("Probability"))
                tr_result.write.save(self.training_result_dest_template.format(cur_of), mode="overwrite")


        elif self.eval_cv_or_tvt == "TVT":
            pipeline = Pipeline(stages=[cur_classifier])
            orig_tr_inst = tr_inst
            orig_te_inst = te_inst
            self.logger.info( "ORIGINAL_INSTANCES")
            #of pop_overview
            from pyspark.sql.functions import count,datediff
            from pyspark.sql.functions import udf,log,sum,exp,max
            udf_prob = udf(lambda x: x.toArray().tolist()[1])
            from pyspark.sql.functions import corr, udf,isnan
            for cur_of in model_of:
                self.logger.debug( cur_of)
                if "{0}_excl".format(cur_of) not in orig_tr_inst.columns:
                    self.logger.info( "NO TARGET {0} is in pts".format(cur_of))
                    continue
                tr_inst = orig_tr_inst.where(col("{0}_excl".format(cur_of)) == 0).withColumn("label",col("{0}_label".format(cur_of)).cast("double")).repartition(500).checkpoint()
                self.logger.info( "Excluded instances for training:{0}".format(orig_tr_inst.where(col("{0}_excl".format(cur_of)) == 1).count()))

                self.logger.info( "TR_POP")
                tr_inst.groupBy("label").agg(count("*")).show()


                te_inst = orig_te_inst.withColumn("label",col("{0}_label".format(cur_of)).cast("double"))

                self.logger.info( "TE_POP")
                te_inst.groupBy("label").agg(count("*")).show()

                tr_inst.printSchema()
                tr_val_pts_dict = self.get_target_tr_val_id()

                tr_pts = tr_val_pts_dict["TR"]
                val_pts = tr_val_pts_dict["VAL"]
                self.logger.info(tr_pts)
                self.logger.info(val_pts)

                orig_tr_inst = tr_inst
                tr_inst = orig_tr_inst.where(col("ID").isin(tr_pts))#.persist()
                val_inst = orig_tr_inst.where(col("ID").isin(val_pts))#.persist()

                tr_inst.show()
                val_inst.show()


                self.logger.info("tr_inst_count:{0}//val_inst_count{1}".format(tr_inst.count(),val_inst.count()))
                te_inst.printSchema()

                # NEED to change with other name, for example pipeline model.
                pipeline_models = pipeline.fit(tr_inst,params=paramGrid)

                max_pr = -1.0
                bestModel = None
                for cur_model in pipeline_models:
                    val_pred = cur_model.transform(val_inst)
                    agg_prob_val = val_pred.groupBy("ID").agg(max("label").alias("label"),sum(log(1.0-udf_prob("Probability"))).alias("inverse_log_sum"))\
                        .select("label",(1.0-exp(col("inverse_log_sum"))).alias("rawPrediction"))
                    agg_prob_val.show(300,truncate=False)
                    cur_pr = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label",metricName=target_metric).evaluate(agg_prob_val)
                    self.logger.info(cur_pr)
                    if max_pr < cur_pr:
                        max_pr = cur_pr
                        bestModel = cur_model

                if not bestModel:
                    self.logger.info("NO MODEL")
                    return
                self.logger.debug( bestModel)
                self.logger.debug(max_pr)
                udf_prob = udf(lambda x: float(x.toArray().tolist()[1]))
                prediction = bestModel.transform(te_inst)
                prediction.show()
                prediction.write.save(self.testing_result_dest_template.format(cur_of), mode="overwrite")
                tr_result = bestModel.transform(tr_inst).withColumn("Prob",udf_prob("Probability"))
                tr_result.write.save(self.training_result_dest_template.format(cur_of), mode="overwrite")
                #tr_inst.show_corr_result(tr_result)
                from pyspark.mllib.evaluation import BinaryClassificationMetrics
                self.logger.info("MAX_PRC_VAL:{0}".format(max_pr))
                # Just Save the Crossvalidation model so that I can access the avgMetric later.
                bestModel.save(self.model_dir_template.format(cur_of,max_pr))
                tr_inst.unpersist()
                val_inst.unpersist()

    @abc.abstractmethod
    def get_param_grid(self,cur_model_selection):
        pass

    def run_experiment(self,num_intv = 5):
        '''
        print agg_prob, corr_predicted_risks and return following
        :return: dict tr_instance df, dict te_instance df, dict model(CVModel)
        '''
        from pyspark.ml import PipelineModel
        import pyspark
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")
        self.set_top_intv_k(num_intv)
        self.flatten_terminal_outcome()
        self.logger.info("experiment_runner in")
        from glob import glob
        tr_result = dict()
        te_result = dict()
        ret_model = dict()
        for cur_of in [self.target_disch_col]:
            try:
                cur_model = glob(self.model_dir_template.format(cur_of,"*"))
                self.logger.info(cur_model)
                if len(cur_model) > 1:
                    self.logger.info( "Exists more than one model. Import following model: {0}".format(cur_model[0]))
                if len(cur_model) == 0:
                    self.logger.info(self.model_dir_template.format(cur_of,"*"))
                    self.logger.info( "NO model exists for {0}!. Will pass".format(cur_of))
                    raise Exception
                print(len(cur_model))
                cur_model = cur_model[0]
                ret_model[cur_of] = PipelineModel.load(cur_model)

                tr_result[cur_of] = self.spark.read.parquet(self.training_result_dest_template.format(cur_of))
                te_result[cur_of] = self.spark.read.parquet(self.testing_result_dest_template.format(cur_of))

            except Exception as ex:
                # TODO erase this after stablized
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.logger.info(message)
                self.logger.info( "No model exists. Training")
                self.define_and_normalize_terminal_df()
                preprocessed_data = self.handle_missing()
                tr_inst, te_inst = self.annotate_dataset(preprocessed_data)
                tr_inst.show()
                te_inst.show()
                self.logger.info("annotated")
                self.run_RF(tr_inst, te_inst,model_of = cur_of)
                self.logger.info("run_RF")

            cur_model = glob(self.model_dir_template.format(cur_of,"*"))
            self.logger.info("AFTER_TRAINING")
            self.logger.info(cur_model)
            print(cur_model)
            if len(cur_model) > 1:
                self.logger.debug( "Exists more than one model. Import following model: {0}".format(cur_model[0]))
            if len(cur_model) == 0:
                self.logger.debug( "NO model exists for {0}!. Will pass".format(cur_of))
                continue
            cur_model = cur_model[0]
            ret_model[cur_of] = PipelineModel.load(cur_model)

            tr_result[cur_of] = self.spark.read.parquet(self.training_result_dest_template.format(cur_of))
            te_result[cur_of] = self.spark.read.parquet(self.testing_result_dest_template.format(cur_of))

            self.logger.info("CUR_OF:{0}".format(cur_of))

        return tr_result, te_result, cur_model


        
        
        


    def flatten_terminal_outcome(self):
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")


        from pyspark.sql.functions import max,col,lit
        terminal_action = self.action_df.select("ID", "ITEMID").distinct()
        #cms_dx_def = self.cur_annotator.annotate_of_label()
        terminal_outcome = self.cur_terminal_df
        from pyspark.sql.functions import when
        cur_of_col = [self.target_disch_col]
        terminal_outcome = terminal_outcome.withColumn(self.target_disch_col,when(col("ICD9_CODE").isin(self.target_disch_icd9),lit("1")).otherwise(lit("0")))\
            .select(["ID"] + cur_of_col).groupBy("ID").agg(
            *(max(c).alias(c) for c in cur_of_col)).select(
            *(col(c).cast("double").alias(c) if c in cur_of_col else col(c) for c in ["ID"] + cur_of_col))
        raw_terminal_outcome = terminal_outcome
        self.logger.debug( "BEFORE_MERGE")
        from pyspark.sql.functions import rand
        #cms_dx_def = self.cur_annotator.annotate_of_label()

        self.target_terminal_outcome_table = terminal_outcome
        flatten_of_list = list()
        for cur_of in [self.target_disch_col]:
            flatten_of_list.append(terminal_outcome.where("{0} == 1".format(cur_of)).select("ID",lit("{0}".format(cur_of)).alias("DISCH_DX")))



        from functools import reduce as f_reduce
        from pyspark.sql import DataFrame
        self.terminal_outcome = f_reduce(DataFrame.union, flatten_of_list).persist()
        return self.terminal_outcome



    def define_and_normalize_terminal_df(self):
        if type(self) == data_run_experiment:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        '''
        exclude action df that doesn't fit to current criteria
        :return:
        '''
        cur_action_col = self.action_df
        cur_def = self.def_df

        original_def = cur_def
        #INPUTEVENT/PROCEDUREEVENTS INCLUSION CRITERIA
        all_itemid =self.get_action_itemids()

        self.action_df = self.action_df.join(all_itemid,["ITEMID","SOURCE"]).persist()

        self.logger.debug( "AF")
        self.action_df.show()

