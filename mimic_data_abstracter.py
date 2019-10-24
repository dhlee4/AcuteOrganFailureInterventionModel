import argparse

from data_abstracter import data_abstracter


class mimic_data_abstracter(data_abstracter):
    def __init__(self,target_env=None, is_debug=True,cur_signature = ""):
        print("INIT IN")
        data_abstracter.__init__(self, target_env, is_debug,cur_signature)
        print("INIT OUT")

    def get_def_df(self):
        import pyspark
        try:
            return self.spark.read.parquet(self.cached_def_df)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")

        from pyspark.sql.functions import lit, col, concat
        cur_chart_df = self.spark.read.parquet(self.data_dir+"/D_ITEMS").where("LINKSTO == 'chartevents'").select("ITEMID","LABEL",lit("VITAL").alias("SOURCE"))

        cur_med_df = self.spark.read.parquet(self.data_dir + "/D_ITEMS").where("LINKSTO == 'inputevents_mv'").select(
            "ITEMID", "LABEL", lit("MED").alias("SOURCE"))

        cur_proc_df = self.spark.read.parquet(self.data_dir + "/D_ITEMS").where("LINKSTO == 'procedureevents_mv'").select(
            "ITEMID", "LABEL", lit("PROC").alias("SOURCE"))

        cur_lab_df = self.spark.read.parquet(self.data_dir+"/D_LABITEMS").select("ITEMID","LABEL",lit("LAB").alias("SOURCE"))

        cur_cpt_df = self.spark.read.parquet(self.data_dir+"/CPTEVENTS").fillna("")\
            .select(col("CPT_CD").alias("ITEMID"),concat("SECTIONHEADER", lit("$$"), "SUBSECTIONHEADER", lit("$$"), "DESCRIPTION").alias("LABEL")
                    ,lit("CPT").alias("SOURCE")).distinct()

        cur_ICD_df = self.spark.read.parquet(self.data_dir+"/D_ICD_DIAGNOSES")\
            .select(col("ICD9_CODE").alias("ITEMID"),col("LONG_TITLE").alias("LABEL"),lit("ICD_DIAGNOSES").alias("SOURCE"))

        def_df = self.spark.sparkContext.union([cur_med_df.rdd, cur_proc_df.rdd\
                                                   , cur_chart_df.rdd, cur_lab_df.rdd, cur_cpt_df.rdd,cur_ICD_df.rdd]).toDF()
        def_df.write.save(self.cached_def_df)
        return def_df


    def remove_dnr_pts(self, cur_chart = None):
        spark = self.spark
        if cur_chart is None:
            cur_chart = self.obs_df

        self.logger.debug(cur_chart.count())

        from pyspark.sql.functions import col,min
        from pyspark.sql.functions import broadcast

        # DNR instances EXCLUDE
        cur_dnr_assertion = ["DNR (do not resuscitate)", "DNR / DNI", "Comfort measures only", "DNI (do not intubate)"]

        cur_dnr_inst = cur_chart.where("ITEMID == 223758").where(col("VALUE").isin(cur_dnr_assertion))\
            .repartition("ID").groupBy("ID").agg(min("TIME_OBS").alias("DNR_TIME"))

        self.logger.info (cur_dnr_inst.count())
        self.logger.info ("BEFORE_DNR_EXCLUDE:{0}".format(cur_chart.count()))
        cur_chart = cur_chart.join(broadcast(cur_dnr_inst), "ID", "left_outer").where(
            (col("DNR_TIME").isNull()) | (col("DNR_TIME").cast("timestamp") > col("TIME_OBS").cast("timestamp")))\
            .checkpoint()
        self.logger.info ("AFTER DNR_EXCLUDE:{0}".format(cur_chart.count()))
        cur_chart = cur_chart.drop("DNR_TIME")

        pts_icu_stay = self.get_icu_stay()
        cur_chart = cur_chart.join(broadcast(pts_icu_stay), "ID").where("INTIME <= TIME_OBS AND OUTTIME>=TIME_OBS").drop(
            "INTIME").drop("OUTTIME").checkpoint()  # .cache()
        self.logger.info(cur_chart.count())
        return cur_chart

    def get_obs_df(self):
        '''
        return dataframe with ID, TIME_OBS, ITEMID, VALUE, SOURCE
        AS IS! without filtering, tagging etc..

        prep dictionary for ITEMID
        :return:
        '''
        import pyspark
        try:
            # TODO remove this show below
            return self.spark.read.parquet(self.cached_obs_df)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")

        from pyspark.sql.functions import col,lit
        cur_lab =self.spark.read.parquet(self.data_dir+"/LABEVENTS")
        cur_vital =self.spark.read.parquet(self.data_dir+"/CHARTEVENTS")

        #might need to change some elements in here to match MIMIC3 dataset.
        cur_lab = cur_lab.select(col("HADM_ID").alias("ID"),col("VALUE"),col("CHARTTIME").alias("TIME_OBS")\
                                 , col("ITEMID"), lit("LAB").alias("SOURCE"))
        cur_vital = cur_vital.select(col("HADM_ID").alias("ID"), col("VALUE"), col("CHARTTIME").alias("TIME_OBS") \
                                 , col("ITEMID"), lit("VITAL").alias("SOURCE"))
        # filtering out the right observation will be done in here

        merged_obs = cur_lab.unionAll(cur_vital)
        merged_obs = merged_obs.join(self.get_icu_stay(),"ID")\
            .where((col("INTIME").cast("timestamp")<=col("TIME_OBS").cast("timestamp")) & (col("TIME_OBS").cast("timestamp") <= col("OUTTIME").cast("timestamp")))\
            .drop("INTIME").drop("OUTTIME")


        merged_obs = self.remove_dnr_pts(cur_chart = merged_obs)

        merged_obs.write.save(self.cached_obs_df)
        return self.spark.read.parquet(self.cached_obs_df)


    def get_action_df(self):
        import pyspark
        from pyspark.sql.functions import col,lit
        try:
            return self.spark.read.parquet(self.cached_action_df)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
        try:
            cur_med =self.spark.read.parquet(self.data_dir+"/INPUTEVENTS_MV")
            cur_med.show()
            cur_med = cur_med.select(col("HADM_ID").alias("ID"), col("STARTTIME").cast("date")\
                                     .cast("string").alias("TIME_OBS"),col("ITEMID"),lit("MED").alias("SOURCE"))
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("NO_INPUT EVENTS")
            cur_med = None
        try:
            cur_cpt =self.spark.read.parquet(self.data_dir+"/CPTEVENTS")\
                .select(col("HADM_ID").alias("ID"),col("CHARTDATE").cast("date")\
                                     .cast("string").alias("TIME_OBS").alias("TIME_OBS"),col("CPT_NUMBER").alias("ITEMID"),lit("CPT").alias("SOURCE"))
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info (message)
            self.logger.info ("NO_CPT EVENTS")
        try:
            cur_proc =self.spark.read.parquet(self.data_dir+"/PROCEDUREEVENTS_MV")\
                .select(col("HADM_ID").alias("ID"),col("STARTTIME").cast("date")\
                                     .cast("string").alias("TIME_OBS"),col("ITEMID"),lit("PROC").alias("SOURCE"))
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info( message)
            self.logger.info( "NO_procedure EVENTS")

        ret_df = None
        for cur_df in [cur_med,cur_cpt,cur_proc]:
            if cur_df:
                if ret_df is not None:
                    ret_df = ret_df.unionAll(cur_df)
                else:
                    ret_df = cur_df
        ret_df.distinct().write.save(self.cached_action_df)
        return self.spark.read.parquet(self.cached_action_df)

    def get_terminal_df(self):
        import pyspark
        try:
            return self.spark.read.parquet(self.cached_terminal_df)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
        from pyspark.sql.functions import col
        cur_df =self.spark.read.parquet(self.data_dir+"\DIAGNOSES_ICD")
        ret_df = cur_df.select(col("HADM_ID").alias("ID"),col("ICD9_CODE"))
        ret_df.write.save(self.cached_terminal_df)
        return ret_df

    def get_action_itemids(self):
        '''
        Maybe need to move this under model abstractor
        :return:
        '''
        # TODO Why disch Dx included in here?
        import pyspark
        try:
            return self.spark.read.parquet(self.temp_action_df)
        except pyspark.sql.utils.AnalysisException as ex:
            self.logger.debug("RUN_PROCESS")

        from pyspark.sql.functions import col

        cur_target_action_cpt_rdd = self.spark.read.parquet(self.data_dir+"/CPTEVENTS") \
            .where(col("CHARTDATE").isNotNull()).where("SECTIONHEADER <> 'Evaluation and management'")\
            .select("CPT_CD").rdd.flatMap(list).map(lambda x: {"SOURCE":"CPT","ITEMID":x})

        cur_target_action_med_rdd = self.spark.read.parquet(self.data_dir+"/D_ITEMS")\
            .where("DBSOURCE == 'metavision' and LINKSTO = 'inputevents_mv'")\
            .where(col("CATEGORY").isin(["Dialysis", "2-Ventilation", "Blood Products/Colloids", "4-Procedures"
                                                        , "1-Intubation/Extubation","3-Significant Events", "Medications"]))\
            .select("ITEMID").rdd.flatMap(list).map(lambda x:{"SOURCE":"MED","ITEMID":x})
        cur_target_action_proc_rdd = self.spark.read.parquet(self.data_dir+"/D_ITEMS")\
            .where("DBSOURCE == 'metavision' and LINKSTO = 'procedureevents_mv'")\
            .where(col("CATEGORY").isin(["Dialysis", "2-Ventilation", "Blood Products/Colloids", "4-Procedures"
                                                        , "1-Intubation/Extubation","3-Significant Events", "Medications"]))\
            .select("ITEMID").rdd.flatMap(list).map(lambda x:{"SOURCE":"PROC","ITEMID":x})

        self.spark.sparkContext.union([cur_target_action_cpt_rdd,cur_target_action_med_rdd,cur_target_action_proc_rdd]).\
            toDF().distinct().write.save(self.temp_action_df)

        return self.spark.read.parquet(self.temp_action_df)

    def get_hospital_death(self):
        from pyspark.sql.functions import col
        return self.spark.read.parquet(self.data_dir+"ADMISSIONS").select(col("HADM_ID").alias("ID"),col("HOSPITAL_EXPIRE_FLAG").alias("IS_DEAD"))

    def get_icu_stay(self):
        '''

        :return: dataframe with ICU stay information, ID, INTIME, OUTTIME
        '''
        from pyspark.sql.functions import col
        return self.spark.read.parquet(self.data_dir+ "ICUSTAYS").where("DBSOURCE == 'metavision'").select(col("HADM_ID").alias("ID"), "INTIME", "OUTTIME")



if __name__ == "__main__":

    import argparse
    cur_target_env = None

    cur_abs = mimic_data_abstracter(target_env = cur_target_env,is_debug=True)
    from spark_and_logger import spark_and_logger
    cur_spark_and_logger =spark_and_logger()
    logger =cur_abs.logger
    cur_abs.def_df.show()
    logger.info("SHOW_OBS_DF")
    cur_abs.obs_df.show()
    logger.info("SHOW_OBS_DF_END")
    logger.info("SHOW_ACTION_DF")
    cur_abs.action_df.show()
    logger.info("SHOW_ACTION_DF_END")
    logger.info("TERMINAL_DF")
    cur_abs.terminal_df.show()
    logger.info("TERMINAL_DF_END")
