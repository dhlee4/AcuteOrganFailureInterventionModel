#import findspark
#findspark.init()
#Be consistent on repartition_const. Either only handle when asserted or only handle when not asserted, not both, not mixed

from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,lit
from spark_and_logger import spark_and_logger


class preprocessor_gen():
    # TODO get rid of sc from the codes. Use self.spark instead
    def __init__(self):
        cur_spark_and_logger = spark_and_logger()
        spark_inited = False
        self.spark = cur_spark_and_logger.spark
        self.logger = cur_spark_and_logger.logger
        if spark_inited:
            self.logger.info("SPARK_INITED")

    def num_cat_tagger(self,data_frame, inputCol="VALUE", outputCol="IS_CAT",labelCol="ITEMID",REPARTITION_CONST = None,nominal_th = 2):
        #DL0411: Output should be list of features with corresponding num/cat instances
        from pyspark.sql.functions import size,collect_set
        if REPARTITION_CONST is None:
            get_nominal_var = data_frame.repartition(labelCol).groupBy(labelCol).agg(size(collect_set(inputCol)).alias("value_cnt"))\
                         .where("value_cnt<={0}".format(nominal_th)).select(labelCol).rdd.flatMap(list).collect()
        else:
            get_nominal_var = data_frame.repartition(REPARTITION_CONST).groupBy(labelCol).agg(size(collect_set(inputCol)).alias("value_cnt"))\
                         .where("value_cnt<={0}".format(nominal_th)).select(labelCol).rdd.flatMap(list).collect()
        self.logger.debug(get_nominal_var)
        data_frame.show()

        self.logger.debug("[PREPROCESSOR_GEN]GET_BINARY_VAR")
        if REPARTITION_CONST is None:
            ret_data_frame = data_frame.withColumn(outputCol,
                when((col(inputCol).rlike('^(?!-0?(\.0+)?(E|$))-?(0|[1-9]\d*)?(\.\d+)?(?<=\d)(E-?(0|[1-9]\d*))?$'))&
                     (~col(labelCol).isin(get_nominal_var)),lit("0"))\
                .otherwise(lit("1")))
        else:
            ret_data_frame = data_frame.withColumn(outputCol,
                when((col(inputCol).rlike('^(?!-0?(\.0+)?(E|$))-?(0|[1-9]\d*)?(\.\d+)?(?<=\d)(E-?(0|[1-9]\d*))?$'))&
                     (~col(labelCol).isin(get_nominal_var)),lit("0"))\
                .otherwise(lit("1"))).repartition(REPARTITION_CONST)
        self.logger.debug("[PREPROCESSOR_GEN]RET_DATA_FRAME")
        return ret_data_frame

    def count_instance(self,raw_feature1, raw_feature2 = None):
        from pyspark.sql.functions import collect_set,size
        raw1_distinct_instance = raw_feature1\
            .select("ID","TIME_SPAN").withColumn("T1",col("TIME_SPAN").TIME_FROM.cast("timestamp"))\
            .withColumn("T2",col("TIME_SPAN").TIME_TO.cast("timestamp")).select("ID","T1","T2")
        if raw_feature2 is not None:
            raw2_distinct_instance = raw_feature2\
                .select("ID","TIME_SPAN").withColumn("T1",col("TIME_SPAN").TIME_FROM.cast("timestamp"))\
                .withColumn("T2",col("TIME_SPAN").TIME_TO.cast("timestamp")).select("ID","T1","T2")
        else:
            raw2_distinct_instance = None
        if raw2_distinct_instance is not None:
            all_inst = raw1_distinct_instance.unionAll(raw2_distinct_instance)
        else:
            all_inst = raw1_distinct_instance

        return all_inst.distinct().groupBy("ID").agg(collect_set("T1").alias("collected_set")).select("ID",size("collected_set").alias("list_size")).rdd.map(lambda x: x.list_size).reduce(lambda a,b:a+b)

    def num_iqr_filter(self,data_frame, inputCol="VALUE",labelCol="ITEMID",REPARTITION_CONST = None,sc=None):
        from pyspark.sql.window import Window
        from pyspark.sql.functions import abs,percent_rank,row_number,collect_list,udf,struct,count,avg,stddev_pop
        from pyspark.sql.types import MapType,StringType,DoubleType
        self.logger.debug("[NUM_IQR_FILTER] IN")
        self.logger.debug("[NUM_IQR_FILTER] BEFORE QUANTILE TAGGED")
        if REPARTITION_CONST is None:
            value_order = Window.partitionBy(labelCol).orderBy(col(inputCol).cast("float"))
            Q1_percentile = Window.partitionBy(labelCol).orderBy(abs(0.25-col("percentile")))
            Q2_percentile = Window.partitionBy(labelCol).orderBy(abs(0.5-col("percentile")))
            Q3_percentile = Window.partitionBy(labelCol).orderBy(abs(0.75-col("percentile")))
            percent_data_frame = data_frame.select(labelCol, inputCol, percent_rank().over(value_order).alias("percentile"))
            Q1_data_frame = percent_data_frame.withColumn("Q1_rn",row_number().over(Q1_percentile)).where("Q1_rn == 1")\
                            .select(labelCol,inputCol,lit("Q1").alias("quantile"))
            Q2_data_frame = percent_data_frame.withColumn("Q2_rn",row_number().over(Q2_percentile)).where("Q2_rn == 1")\
                            .select(labelCol,inputCol,lit("Q2").alias("quantile"))
            Q3_data_frame = percent_data_frame.withColumn("Q3_rn",row_number().over(Q3_percentile)).where("Q3_rn == 1")\
                            .select(labelCol,inputCol,lit("Q3").alias("quantile")) # why divide?
            self.logger.debug("[NUM_IQR_FILTER] REPARTITION_CONST Not Asserted")
            merge_all = Q1_data_frame.unionAll(Q2_data_frame).unionAll(Q3_data_frame).persist()

            self.logger.debug("[NUM_IQR_FILTER] Qs Merged")

            #debug_purpose
            udf_parse_list_to_map = udf(lambda maps: dict(list(tuple(x) for x in maps)),MapType(StringType(),StringType()))

            self.logger.debug("[NUM_IQR_FILTER] Before merge quantiles")
            aggregate_quantiles = merge_all.groupBy(labelCol).agg(collect_list(struct("quantile",inputCol)).alias("quant_info"))
            #aggregate_quantiles.show()
            self.logger.debug("AGG ONLY")
            aggregate_quantiles = aggregate_quantiles.select(labelCol,udf_parse_list_to_map("quant_info").alias("quant_info"))
            #aggregate_quantiles.show()
            self.logger.debug("TRANSFORM")
            iqr_data_frame = aggregate_quantiles.withColumn("Q1",col("quant_info").getItem("Q1").cast("float"))\
                .withColumn("Q2",col("quant_info").getItem("Q2").cast("float"))\
                .withColumn("Q3",col("quant_info").getItem("Q3").cast("float"))
            #debug_purpose
            #aggregate_quantiles.show()
            self.logger.debug("QUANTILE_EXTRACTION")
        else:
            cur_label_list = data_frame.select(labelCol).distinct().rdd.flatMap(list).collect()
            cur_iqr_list = list()
            cnt = -1
            for cur_item in cur_label_list:
                cnt = cnt+1
                data_frame.where(col(labelCol) == cur_item).registerTempTable("cur_table")
                self.logger.debug("{0}/{1},::{2}".format(cnt,len(cur_label_list),sc.sql("select {0} from cur_table".format(labelCol)).count()))
                cur_iqr = sc.sql("select {0}, percentile_approx({1},0.25) as Q1, percentile_approx({2},0.5) as Q2, percentile_approx({3},0.75) as Q3 from cur_table group by {4}".format(labelCol,inputCol,inputCol,inputCol,labelCol)).first().asDict()
                cur_iqr_list.append(cur_iqr)
                sc.catalog.dropTempView("cur_table")
                #percent_data_frame = data_frame.select(labelCol, inputCol, percent_rank().over(value_order).alias("percentile")).repartition(REPARTITION_CONST).cache().checkpoint()
            iqr_data_frame = sc.createDataFrame(cur_iqr_list).repartition(REPARTITION_CONST)


        if REPARTITION_CONST is None:
            iqr_data_frame = iqr_data_frame.withColumn("IQR",col("Q3")-col("Q1"))\
                                       .withColumn("LB",col("Q1")-1.5*col("IQR"))\
                                       .withColumn("UB",col("Q3")+1.5*col("IQR"))\
                                       .select(labelCol,"LB","UB")
        else:
            iqr_data_frame = iqr_data_frame.withColumn("IQR",col("Q3")-col("Q1"))\
                                       .withColumn("LB",col("Q1")-1.5*col("IQR"))\
                                       .withColumn("UB",col("Q3")+1.5*col("IQR"))\
                                       .select(labelCol,"LB","UB").repartition(REPARTITION_CONST).persist()

            self.logger.debug("CUR_ITEMID_ALL_COUNT:{0}".format(iqr_data_frame.count()))

        self.logger.debug("[NUM_IQR_FILTER] iqr_data_frame merged")
        if REPARTITION_CONST is None:
    #        data_frame.show()
    #        iqr_data_frame.show()
            self.logger.debug("[NUM_IQR_FILTER] RETURN_PREP, REPARTITION_CONST NOT ASSERTED")
            ret_data_frame = data_frame.repartition(labelCol).join(iqr_data_frame,labelCol).where((col("LB").cast("float") <= col(inputCol).cast("float")) & (col("UB").cast("float")>=col(inputCol).cast("float")))\
                                                                 .drop("LB").drop("UB").persist()
            ref_df = ret_data_frame.repartition(labelCol).groupBy(labelCol)\
                               .agg(count(inputCol).alias("ref_count"),avg(inputCol).alias("ref_avg"),stddev_pop(inputCol).alias("ref_std")).persist()
            self.logger.debug("CHECK DF")
            self.logger.debug(ref_df.count())
            self.logger.debug( ret_data_frame.count())

            return (ret_data_frame, ref_df)
        else:
            self.logger.debug("[NUM_IQR_FILTER] RETURN_PREP, REPARTITION_CONST ASSERTED: {0}".format(REPARTITION_CONST))
            ret_data_frame = data_frame.join(iqr_data_frame,labelCol).where((col("LB").cast("float") <= col(inputCol).cast("float")) & (col("UB").cast("float")>=col(inputCol).cast("float")))\
                                                                 .drop("LB").drop("UB").repartition(REPARTITION_CONST)
            ref_df = ret_data_frame.groupBy(labelCol)\
                               .agg(count(inputCol).alias("ref_count"),avg(inputCol).alias("ref_avg"),stddev_pop(inputCol).alias("ref_std")).repartition(REPARTITION_CONST)

            return (ret_data_frame, ref_df)

    def cat_frequency_filter(self,data_frame,threshold_lb = 0,threshold_ub = 1,inputCol="VALUE",labelCol="ITEMID",REPARTITION_CONST=None):
        from pyspark.sql.functions import count,monotonically_increasing_id
        label_count = data_frame.groupBy(labelCol).agg(count("*").alias("label_count"))
        if REPARTITION_CONST is not None:
            label_count = label_count.repartition(REPARTITION_CONST).persist()
        self.logger.debug("[CAT_FREQUENCY_FILTER] label_count done]")
        if REPARTITION_CONST is None:
            cur_freq = data_frame.groupBy(labelCol,inputCol).agg(count("*").alias("indiv_count")).join(label_count,labelCol)\
                                .withColumn("cat_freq",col("indiv_count")/col("label_count"))
        else:
            cur_freq = data_frame.groupBy(labelCol,inputCol).agg(count("*").alias("indiv_count")).repartition(REPARTITION_CONST).join(label_count,labelCol)\
                                .withColumn("cat_freq",col("indiv_count")/col("label_count")).repartition(REPARTITION_CONST).checkpoint()
            self.logger.debug("[CAT_FREQUENCY_FILTER] CUR_FREQ_CHECKPOINTED:{0}".format(cur_freq.count()))
        self.logger.debug("[CAT_FREQUENCY_FILTER] frequency calc done")
        cur_freq = cur_freq.where((col("cat_freq")>=threshold_lb) & (col("cat_freq")<=threshold_ub)).drop("cat_freq")
        #cur_freq.orderBy(col(labelCol),-1*col("cat_freq")).show(500)

        ret_data_frame = data_frame.join(cur_freq,[inputCol,labelCol]).drop("indiv_count").drop("label_count")
        self.logger.debug("[CAT_FREQUENCY_FILTER] ret_df prepared")
        if REPARTITION_CONST is not None:
            ret_data_frame = ret_data_frame.repartition(REPARTITION_CONST)
        ret_voca = ret_data_frame.select("ITEMID","VALUE").distinct()#.withColumn("idx",monotonically_increasing_id())
        ret_voca = ret_voca.rdd.map(lambda x: (x.ITEMID,x.VALUE)).zipWithUniqueId().map(lambda x: {"idx":x[1], "ITEMID":x[0][0], "VALUE":x[0][1]}).toDF()
        if REPARTITION_CONST is not None:
            ret_voca = ret_voca.repartition(REPARTITION_CONST)
        self.logger.debug("[CAT_FREQUENCY_FILTER] ALL DONE")
        return (ret_data_frame, ret_voca)

    @staticmethod
    def calc_summary_stat(x,labelCol):
        import numpy as np
        cur_array = np.array(x,dtype=float)
        ret_dict = dict()
        ret_dict["N_{0}_avg".format(labelCol)] = float(np.average(cur_array))
        ret_dict["N_{0}_min".format(labelCol)] = float(np.min(cur_array))
        ret_dict["N_{0}_max".format(labelCol)] = float(np.max(cur_array))
        ret_dict["N_{0}_std".format(labelCol)] = float(np.std(cur_array))
        ret_dict["N_{0}_count".format(labelCol)] = float(np.shape(cur_array)[0])
        return ret_dict
    @staticmethod
    def merge_dict_all(x): #will not be used outside of the package
        ret_dict = dict()
        for cur_dict in x:
            ret_dict.update(cur_dict)
        return ret_dict

    @staticmethod
    def sustainment_quantifier(x,cur_label,ref_avg, ref_std, ref_count):
        from scipy.stats import ttest_ind_from_stats
        import numpy as np
        ret_dict = dict()
        statistic, p_val = ttest_ind_from_stats(x["N_{0}_avg".format(cur_label)], x["N_{0}_std".format(cur_label)], x["N_{0}_count".format(cur_label)], ref_avg, ref_std, ref_count, equal_var=True)
        if not np.isnan(statistic):
            if statistic > 0:
                one_tailed_pval = 1.0 - p_val/2.0
            else:
                one_tailed_pval = p_val/2.0
            ret_dict["N_{0}_TT".format(cur_label)] = float(p_val)
            ret_dict["N_{0}_LT".format(cur_label)] = float(one_tailed_pval)
        return ret_dict

    def num_featurizer(self,data_frame, ref_df=None, featurize_process = ["summary_stat","sustainment_q"], inputCol="VALUE",labelCol="ITEMID",outputCol="num_features",REPARTITION_CONST=None):
        from pyspark.sql.functions import udf,array
        from pyspark.sql.types import StringType, DoubleType, MapType
        if not data_frame:
            return
        ret_data_frame = self.value_aggregator(data_frame)
        if REPARTITION_CONST is not None:
            ret_data_frame = ret_data_frame.checkpoint()
            self.logger.debug("[NUM_FEATURIZER] ret_dataframe checkpointed:{0}".format(ret_data_frame.count()))
        if "summary_stat" in featurize_process:
            udf_summary_stat = udf(preprocessor_gen.calc_summary_stat,MapType(StringType(),DoubleType()))
            ret_data_frame = ret_data_frame.withColumn("summary_stat",udf_summary_stat(inputCol+"_LIST",labelCol))
            if REPARTITION_CONST is not None:
                ret_data_frame = ret_data_frame.checkpoint()
                self.logger.debug("[NUM_FEATURIZER] summary_stat, ret_dataframe checkpointed:{0}".format(ret_data_frame.count()))

        if "sustainment_q" in featurize_process:
            udf_sustainment_quant = udf(preprocessor_gen.sustainment_quantifier,MapType(StringType(),DoubleType()))
            ret_data_frame = ret_data_frame.join(ref_df,labelCol).withColumn("sustainment_q",udf_sustainment_quant("summary_stat",labelCol,"ref_avg","ref_std","ref_count")).drop("ref_avg").drop("ref_std").drop("ref_count")
            if REPARTITION_CONST is not None:
                ret_data_frame = ret_data_frame.checkpoint()
                self.logger.debug("[NUM_FEATURIZER] sustainment_q, ret_dataframe checkpointed:{0}".format(ret_data_frame.count()))
        udf_merge_dict_all = udf(preprocessor_gen.merge_dict_all,MapType(StringType(),DoubleType()))
        ret_data_frame = ret_data_frame.withColumn(outputCol,udf_merge_dict_all(array(featurize_process)))
        return ret_data_frame

    def cat_featurizer(self,data_frame, voca_df, inputCol="VALUE",labelCol="ITEMID",outputCol="cat_features",REPARTITION_CONST = None):
        # TODO add test routine for value_aggregator
        def prep_cat_dict(avail, pos):  # internal
            ret_dict_key = list(map(lambda x: "C_" + str(x), avail))
            ret_dict = dict(zip(ret_dict_key, [0.0] * len(ret_dict_key)))
            pos_dict_key = list(map(lambda x: "C_" + str(x), pos))
            update_dict = dict(zip(pos_dict_key, [1.0] * len(pos_dict_key)))
            ret_dict.update(update_dict)
            return ret_dict

        from pyspark.sql.functions import udf,collect_set
        from pyspark.sql.types import MapType, StringType, DoubleType
        if not data_frame:
            return

        all_var = voca_df.groupBy(labelCol).agg(collect_set("idx").alias("AVAIL_LIST"))
        ret_data_frame = data_frame.join(voca_df,[inputCol,labelCol]).drop("VALUE").withColumnRenamed("idx","VALUE")
        if REPARTITION_CONST is not None:
            ret_data_frame = ret_data_frame.repartition(REPARTITION_CONST).checkpoint()
            self.logger.debug("[CAT_FEATURIZER] VOCA_JOINED_CHECKPOINTED:{0}".format(ret_data_frame.count()))
        ret_data_frame = self.value_aggregator(ret_data_frame).join(all_var,"ITEMID")
        self.logger.debug(ret_data_frame.select(labelCol).distinct().count())
        self.logger.debug("[CAT_FEATURIZER] VALUE_AGGREGATOR OUT")
        udf_prep_cat_dict = udf(prep_cat_dict, MapType(StringType(),DoubleType()))
        ret_data_frame = ret_data_frame.withColumn("cat_features",udf_prep_cat_dict("AVAIL_LIST","VALUE_LIST"))
        return ret_data_frame

    def availability_filter(self,data_frame,n_inst = None, availability_th = 0.80,labelCol="ITEMID",idCol="ID",timeCol="TIME_SPAN",REPARTITION_CONST=None):
        from pyspark.sql.functions import count
        if not n_inst:
            total_cnt = data_frame.select(idCol,timeCol).distinct().count()
            self.logger.debug(total_cnt)
        else:
            total_cnt = n_inst
        target_label_set = data_frame.select(idCol,timeCol,labelCol).distinct()

        if REPARTITION_CONST is None:
            target_label_set = target_label_set.groupBy(labelCol).agg((count("*")/float(total_cnt)).alias("freq"))
        else:
            target_label_set = target_label_set.repartition(REPARTITION_CONST)\
                .groupBy(labelCol).agg((count("*")/float(total_cnt)).alias("freq"))

        target_label_set.orderBy(col("freq").desc()).show()

        if REPARTITION_CONST is not None:
            target_label_set = target_label_set.repartition(REPARTITION_CONST).checkpoint()
            self.logger.info("[AVAILABILITY_FILTER] target_label_Set checkpointed:{0}".format(target_label_set.count()))
        self.logger.info(target_label_set.rdd.toDebugString())
        target_label_set = target_label_set.where(col("freq")>=availability_th).select(labelCol).rdd.flatMap(list).collect()
        self.logger.info(target_label_set)
        if len(target_label_set) == 0:
            return
        ret_data_frame = data_frame.where(col(labelCol).isin(target_label_set))

        #fordebug
        self.logger.debug(target_label_set)
        self.logger.debug(len(target_label_set))
        return ret_data_frame

    @staticmethod
    def check_key_in_dict(target_dict,target_key):
        return target_dict.has_key(target_key)

    def flattener_df_prep(self,data_frame, descCol=["ID","TIME_SPAN"] ,inputCol="feature_aggregated",drop_cnt=True):
        from pyspark.sql import Row
        from pyspark.sql.functions import col,udf,lit
        from pyspark.sql.types import StructType,StructField,StringType,DoubleType,BooleanType
        data_frame.show()
        desc_schema = data_frame.select(descCol).schema
        all_feature_column = list(data_frame.select(inputCol).rdd.map(lambda x: set(x[inputCol].keys())).reduce(lambda a,b: a.union(b)))
        ret_df = data_frame.rdd.map(lambda x: x.asDict()).map(lambda cur_item: [dict((cur_col,cur_item[cur_col]) for cur_col in descCol)]\
                                                                                + [cur_item[inputCol]]).map(lambda x: preprocessor_gen.merge_dict_all(x))
        inst_count = data_frame.count()
        ret_schema = desc_schema
        ret_feature_col = list()
        key_checker = udf(preprocessor_gen.check_key_in_dict,BooleanType())
        for cur_col in all_feature_column:
    #        print ("{0}//{1}//{2}".format(data_frame.where(key_checker(inputCol,lit(cur_col))).count(),inst_count,cur_col))
    #        if (data_frame.where(key_checker(inputCol,lit(cur_col))).count() == inst_count):
    #            continue
            if drop_cnt:
                if cur_col.find("count") != -1:
                    continue
            ret_feature_col.append(cur_col)
            ret_schema = ret_schema.add(StructField(cur_col,DoubleType(),True))
        return (ret_df, ret_schema,ret_feature_col)

    #DL0411: imputer will be depreciated from here. It will be the part of data merger
    '''def imputer(data_frame, impute_method="mean",inputCols="features", outputCol="features_imputed"):
    
        from pyspark.ml.feature import Imputer,VectorAssembler
        
        if type(inputCols) != list:
            inputCols = [inputCols]
        imputedCols = list()
        numericalCols = list()
        categoricalCols = list()
        for i in inputCols:
            if i.find("N_") != -1:
                numericalCols.append(i)
            elif i.find("C_") != -1:
                categoricalCols.append(i)
    
        print(categoricalCols)
        print(numericalCols)
        data_frame = data_frame.fillna(0,subset = categoricalCols)
        
        imputedCols = ["imp_{0}".format(x) for x in numericalCols]
    
        print("IMPUTER!!")
        imputer = Imputer(inputCols = numericalCols, outputCols=imputedCols).setStrategy(impute_method)
        imputer_model = imputer.fit(data_frame)
        ret_data_frame = imputer_model.transform(data_frame)
        ret_data_frame.show()
    
        ret_data_frame=VectorAssembler(inputCols=imputedCols+categoricalCols, outputCol=outputCol).transform(ret_data_frame)
        #raised error somewhere near here.
        return (ret_data_frame,imputedCols+categoricalCols)
    '''

    def value_aggregator(self,data_frame, aggregateCols = ["ID","TIME_SPAN","ITEMID"], catmarkerCol="IS_CAT", inputCol = "VALUE", outputCol="VALUE_LIST"):
        from pyspark.sql.functions import collect_set, collect_list
        cat_data_agg_frame = data_frame.where(col(catmarkerCol) == 1).groupBy(aggregateCols+[catmarkerCol])\
                                                                     .agg(collect_set(inputCol).alias(outputCol))
        num_data_agg_frame = data_frame.where(col(catmarkerCol) == 0).groupBy(aggregateCols+[catmarkerCol])\
                                                                     .agg(collect_list(inputCol).alias(outputCol))
        return cat_data_agg_frame.unionAll(num_data_agg_frame)

    def feature_aggregator(self,num_features,cat_features,catinputCol="cat_features",numinputCol="num_features",aggregatorCol = ["ID","TIME_SPAN"], outputCol="feature_aggregated",idCol="ID",REPARTITION_CONST = None):
        from pyspark.sql.functions import col, udf, collect_list,rand
        from pyspark.sql.types import MapType, StringType, DoubleType

        if not num_features:
            ret_data_frame=cat_features.withColumnRenamed(catinputCol,"features")
            self.logger.debug("CAT_ONLY")
        elif not cat_features:
            ret_data_frame=num_features.withColumnRenamed(numinputCol,"features")
            self.logger.debug("NUM_ONLY")
        else:
            ret_data_frame = num_features.select(aggregatorCol+[numinputCol]).withColumnRenamed(numinputCol,"features")\
                                     .unionAll(cat_features.select(aggregatorCol+[catinputCol]).withColumnRenamed(catinputCol,"features"))
            self.logger.debug("BOTH")
        if REPARTITION_CONST is not None:
            ret_data_frame = ret_data_frame.repartition(REPARTITION_CONST).checkpoint()
            self.logger.debug("[FEATURE_AGGREGATOR] ret_data_Frame checkpointed before groupby:{0}".format(ret_data_frame.count()))
        udf_merge_dict_all = udf(preprocessor_gen.merge_dict_all,MapType(StringType(),DoubleType()))
        self.logger.debug("[FEATURE_AGGREGATOR] before groupby")
        # FROM HERE
        ret_data_frame = ret_data_frame.groupBy(aggregatorCol)
        if REPARTITION_CONST is not None:
            ret_data_frame = ret_data_frame.agg(collect_list("features").alias("features")).repartition(REPARTITION_CONST).checkpoint()
            self.logger.debug("[FEATURE_AGGREGATOR] ret_data_frame chkpointed:{0}".format(ret_data_frame.count()))
        else:
            ret_data_frame = ret_data_frame.agg(collect_list("features").alias("features"))
        #ret_data_frame.orderBy(rand()).show(truncate=False)
        ret_data_frame = ret_data_frame.withColumn(outputCol, udf_merge_dict_all("features"))
        return ret_data_frame
        # TILL HERE. Maybe something triggers this weird spark burst

    def post_processor(data_frame, algo="None",outputCol="features_postprocessed"):
        ret_data_frame = data_frame
        return ret_data_frame

    def normalizer(data_frame, inputCol="features", outputCol="scaled_features"):
        ret_data_frame = data_frame
        return ret_data_frame

    def prep_TR_TE(merged_df, per_instance=False, tr_prop=0.9,targetCol="ID"):
        from pyspark.sql.functions import col
        if per_instance:
            tr_inst, te_inst = merged_df.randomSplit([tr_prop, 1-tr_prop])
        else:
            tr_id, te_id = merged_df.select(targetCol).distinct().randomSplit([tr_prop,1-tr_prop])
            tr_id = tr_id.rdd.flatMap(list).collect()
            te_id = te_id.rdd.flatMap(list).collect()
            tr_inst = merged_df.where(col(targetCol).isin(tr_id))
            te_inst = merged_df.where(col(targetCol).isin(te_id))
        return (tr_inst,te_inst)








    # NEED TO GET RID OF THIS PART. NEVER USE FOR THE MAIN PROCESSING!#
    # Will get this rid of as soon as dev done
    if __name__ == "__main__":
        # TODO remove all this information
        cur_dir = "/Users/dhlee4/mimic3_data/CHARTEVENTS.csv_parquet"
        cur_home = "/Users/dhlee4/mimic3_data/"

        spark = SparkSession.builder.master("local[*]")\
                                     .appName("PreProcessorGen_local_test")\
                                     .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        cur_df = spark.read.parquet(cur_dir+"_test_pts")

        prep_df = cur_df.select(col("HADM_ID").alias("ID"), col("CHARTTIME").alias("TIME_OBS")\
                              ,"ITEMID",col("VALUE"))
        from pyspark.sql.functions import struct,split,date_add
        prep_df = prep_df.withColumn("TIME_SPAN",struct(split("TIME_OBS","\ ").getItem(0).cast("timestamp").alias("FROM_TIME")\
                                                        ,date_add(split("TIME_OBS","\ ").getItem(0),1).cast("timestamp").alias("TO_TIME")))
        prep_df.show()
        self.logger.debug(prep_df.count())
        zz = num_cat_tagger(prep_df)
        zz.show(truncate=False)
        cat_raw_filtered, voca_list = cat_frequency_filter(zz.where("IS_CAT == 1"))
        num_raw_filtered, num_ref_list = num_iqr_filter(zz.where("IS_CAT == 0"))
        cur_id = cat_raw_filtered.select("ID","TIME_SPAN").unionAll(num_raw_filtered.select("ID","TIME_SPAN")).distinct()
        num_filtered = availability_filter(num_raw_filtered,cur_id.count())
        num_filtered.show()
        cat_filtered = availability_filter(cat_raw_filtered,cur_id.count())
        cat_filtered.show()

        cat_featurized = cat_featurizer(cat_filtered,voca_df = voca_list)
        num_featurized = num_featurizer(num_filtered,ref_df = num_ref_list)

        merged_all = feature_aggregator(num_featurized, cat_featurized)


        target_rdd, target_schema, feature_column = flattener_df_prep(merged_all)

        cur_df = spark.createDataFrame(target_rdd,target_schema)
        cur_df.show(50)


        imputed_df,feature_col = imputer(cur_df, inputCols=feature_column)
        imputed_df.show(truncate=False)
        self.logger.debug(feature_col)
        imputed_df.write.save(cur_home+"test_obs",mode="overwrite")
