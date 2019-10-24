import abc
from data_abstracter import data_abstracter


class data_preprocessor():
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        from datetime import datetime
        #now it is getting weird
        from preprocessor_gen import preprocessor_gen
        self.cur_preprocessor = preprocessor_gen()
        return

    @abc.abstractmethod
    def remove_dnr_pts(self):
        '''
        remove dnr pts from dataframe
        :return:
        '''
        pass

    def run_preprocessor(self):
        if type(self) == data_preprocessor:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        '''

        #DL0411: Params that needs to be exposed: availability th. spark context? repartition_const, cat_voca_list
        #Useful to output: ref_list. feature columns, processed df
        #spec for processed DFs: missing values will be presented as nulls. numeric values will be noted as N and categorical values will be noted as C.
        #test_id = for test instances
        #something like -- num_summary, cat_summary, feature_columns(in order), processed_df = run_preprocess(0.7,repartition_const,test_id,verbose=True)
        #make sure parquet doesn't change their column orders - NO.https://stackoverflow.com/questions/32748478/is-there-a-possibility-to-keep-column-order-when-reading-parquet?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        #separate the descriptor columns, and let feature columns to ascending orders. make sure this steps in the future work. expected orders will be 1) descriptor columns, 2) categorical variables, 3) numerical variables
        #appending demographic variables is up to classifiers as it depends on the experiment during preprocessing step. Maybe include preprocessing step in here though? Maybe separate them and handle preprocess in here.
        # SOMETHING LIKE run_preprocess_time_series(0.7,repartition_const, test_id, verbose=True)
        #           AND  run_preprocess_demographics(repartition_const,verbose=True)
        #normalization is also not part of the responsibility of this code. it assume the classifiers to do the normalization before the training(if needed)
        #consider repartition original clinical datasets and resave them
        #DL0411: end of comments
        # There are some issues in categorical variables. Low availability features are introduced into the model. Need to figure out why and prepare testcode for validating this. Not only for categorical features, but also numerical features.
        # should parameterize the whole running process. Generate outputs with time stamps. In case we need to rerun it, it should be easily trackable so that we don't need to run whole pipeline. -- DONE
        # also, validate by load dataframe, extract columns and compare with voca

        :return spark dataframe witih preprocessed. NAME: ID,TIME_SPAN// Others - features in OBS:
        '''
        from pyspark.sql.functions import struct, col, split, date_add
        try:
            return self.spark.read.parquet(self.out_file_name)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
            self.logger.info(self.out_file_name)


        try:
            return self.spark.read.parquet(self.out_file_name)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS_PREPROCESS")
            self.logger.info(self.out_file_name)
            

        self.cur_obs_bf_dropna = self.obs_df.select("ID", "ITEMID", "VALUE", "TIME_OBS",
                                          split("TIME_OBS", "\ ").getItem(0).alias("DATE_OBS")) \
            .withColumn("TIME_SPAN", struct(col("DATE_OBS").cast("timestamp").alias("TIME_FROM") \
                                            , date_add("DATE_OBS", 1).cast("timestamp").alias(
                "TIME_TO")))


        self.cur_obs = self.cur_obs_bf_dropna.dropna()



        self.run_remaining()
        return self.spark.read.parquet(self.out_file_name)

    def run_remaining(self, cur_obs = None):
        if type(self) == data_preprocessor:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        '''

         this also don't have to be changed. Additional features i.e., demo. will be appended after the original features.
         it would be easier to match features in the worst case scenario
        :return:
        '''

        if cur_obs is None:
            cur_obs = self.cur_obs

        from pyspark.sql.functions import col,split,struct

        #FOr DEBUG

        self.logger.debug(cur_obs.count())
        cur_obs.show()
        from pyspark.sql.functions import count,col
        cur_obs.groupBy("ITEMID").agg(count("*").alias("cnt")).orderBy(col("cnt").desc()).show(300)

        num_cat_tagged = self.cur_preprocessor.num_cat_tagger(cur_obs)
        # this should only be using TR data. When it is done, num/cat will be tagged next to the testing datase
        cat_raw_filtered, cat_voca_list = self.cur_preprocessor.cat_frequency_filter(num_cat_tagged.where("IS_CAT == 1"))
        cat_voca_list = self.save_voca(cat_voca_list)
        self.logger.info("NUMBERS")
        num_raw_filtered, num_ref_list = self.cur_preprocessor.num_iqr_filter(num_cat_tagged.where("IS_CAT == 0"))
        # If checkpoint fails, it means there is no checkpoint dir set, then persist instead

        try:
            self.logger.debug("CHECKPOINT EXISTS?")
            REPARTITION_CONST=200
            cat_raw_filtered = cat_raw_filtered.checkpoint()
            num_raw_filtered = num_raw_filtered.checkpoint()
        except:
            self.logger.debug("OR_NOT")
            REPARTITION_CONST = None
            cat_raw_filtered = cat_raw_filtered.persist()
            num_raw_filtered = num_raw_filtered.persist()

        self.logger.info("FILTER_DONE")
        num_instance = self.cur_preprocessor.count_instance(cat_raw_filtered, num_raw_filtered)
        self.logger.debug("ALL INSTANCE COUNT:{0}".format(num_instance))

        num_raw_filtered.show()
        cat_raw_filtered.show()

        for cur_th in self.th_range:
            self.logger.info(cur_th)
            cat_filtered = self.cur_preprocessor.availability_filter(cat_raw_filtered, num_instance, availability_th=cur_th,
                                               REPARTITION_CONST=REPARTITION_CONST)
            if cat_filtered is None:
                self.logger.info("NO CAT FEATURES")
            else:
                self.logger.info(cat_filtered.select("ITEMID").distinct().count())
                self.logger.info("CAT_FILTERED")
            
            num_filtered = self.cur_preprocessor.availability_filter(num_raw_filtered, num_instance, availability_th=cur_th,
                                               REPARTITION_CONST=REPARTITION_CONST)
            if num_filtered is None:
                self.logger.info("NO NUM FEATURES")
            else:
                self.logger.info(num_filtered.select("ITEMID").distinct().count())
                self.logger.info("Num_FILTERED")
            self.logger.info("AVAIL_FILTER_DONE")

            cat_featurized = self.cur_preprocessor.cat_featurizer(cat_filtered, voca_df=cat_voca_list\
                                                                  , REPARTITION_CONST=REPARTITION_CONST)
            # stuffs that needs to be matched : unique_n(ITEMID) in cat_featurized, unique_n(ITEMID) in cat_featurized
            # voca should cover all C_[X] instances in preprocessed_data
            #
            self.logger.info("CAT_FEATURIZED")

            num_featurized = self.cur_preprocessor.num_featurizer(num_filtered, ref_df=num_ref_list\
                                                                  , REPARTITION_CONST=REPARTITION_CONST)
            self.logger.info("NUM_FEATURIZED")
            self.logger.info("FEATURIZER_DONE")

            try:
                num_featurized = num_featurized.checkpoint()
                cat_featurized = cat_featurized.checkpoint()
                self.logger.info("NUM_FEATURIZED_CNT:{0}".format(num_featurized.count()))
                try:
                    self.logger.info("CAT_FEATURIZED_CNT:{0}".format(cat_featurized.count()))
                except:
                    # TODO NEED to figure out which error this need to handle
                    self.logger.debug("NO")
            except:
                self.logger.info("NO_CHECKPOINT")

            if (num_featurized is None) and (cat_featurized is None):
                raise Exception("NO FEATURE SELECTED. HALTING THE PROCESS")
            merged_all = self.cur_preprocessor.feature_aggregator(num_featurized, cat_featurized\
                                                                  , REPARTITION_CONST=REPARTITION_CONST)
            target_rdd, target_schema, feature_column = self.cur_preprocessor.flattener_df_prep(merged_all)
            cur_df = self.spark.createDataFrame(target_rdd, target_schema).persist()
            self.logger.info("REMAINING_OUT")
            cur_df.show()
            import time
            time.sleep(30)
            cur_df.write.save(self.out_file_name, mode="overwrite")
            self.logger.info(self.out_file_name)
            self.logger.info("AND_SAVED")
        return cur_df

    def save_voca(self, cat_voca_list):
        if type(self) == data_preprocessor:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        cat_voca_list.write.save(self.voca_name, mode="overwrite")
        return self.spark.read.parquet(self.voca_name)
