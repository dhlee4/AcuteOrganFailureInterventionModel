import abc


class data_source_mask(object):
    def __init__(self,target_env = None, cur_signature = "0603",is_debug = False,dataset_dir="./mimic3_demo_dataset/"):
        '''

        :param target_env: specify target running environment, default="TEST", current implementation=["TEST","HYAK_IKT","TEST","TEST_MAC_PRO"]
        '''
        import logging
        from spark_and_logger import spark_and_logger

        cur_spark_and_logger = spark_and_logger()
        first_run_flag = False
        # TODO need to inherit data_dir or home_dir and drop logs under there
        if cur_spark_and_logger.logger is None:
            first_run_flag = True
            import logging
            logger = logging.getLogger('root')
            FORMAT = "[%(asctime)s%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
            logging.basicConfig(format=FORMAT)
            fileHandler = logging.FileHandler("{0}/{1}.log".format("./", "cur_logger.txt"))
            fileHandler.setFormatter(FORMAT)
            logger.addHandler(fileHandler)
            logger.setLevel(logging.DEBUG)
            cur_spark_and_logger.logger = logger
        self.logger = cur_spark_and_logger.logger


        self.is_debug = is_debug
        self.cur_signature = cur_signature
        self.data_dir = dataset_dir
        import os
        if not os.path.isdir("./output"):
            os.mkdir("./output")
        self.home_dir = "./output/"

        self.home_dir = self.home_dir + self.cur_signature + "/"
        cur_spark_and_logger = spark_and_logger()
        cur_spark_and_logger.home_dir = self.home_dir
        self.intermediate_dir = self.home_dir + "/temp/"
        self.cached_obs_df = "{0}/{1}".format(self.intermediate_dir,"obs_df_intermediate")
        self.cached_action_df = "{0}/{1}".format(self.intermediate_dir, "action_df_intermediate")
        self.cached_terminal_df = "{0}/{1}".format(self.intermediate_dir, "terminal_df_intermediate")
        self.cached_def_df = "{0}/{1}".format(self.intermediate_dir, "def_df_intermediate")
        self.temp_target_id_list_file = self.intermediate_dir + "/target_id_list.txt"
        self.temp_action_df = self.intermediate_dir + "/target_action"
        self.temp_target_test_id_list_file = self.intermediate_dir+"/target_test_id_list.txt"
        self.temp_target_tr_val_id_list_file = self.intermediate_dir+"/target_tr_val_id_list.txt"

        if first_run_flag:
            self.logger.info('HOME_DIR:{0}'.format(self.home_dir))
            self.logger.info('DATA_DIR:{0}'.format(self.data_dir))
            self.logger.info('INTERMEDIATE_DIR:{0}'.format(self.intermediate_dir))



class data_abstracter(data_source_mask):
    __metaclass__ = abc.ABCMeta
    def __init__(self,target_env=None,is_debug=False,cur_signature =""):
        '''

        :param data_source: Init using data_abstracter. Default is TEST
        :param is_debug: Specify whether it is debug mode(take 50 samples from total dataset) or not. Default is debug
        '''
        from spark_and_logger import spark_and_logger
        super(data_abstracter,self).__init__(target_env=target_env,is_debug=is_debug,cur_signature=cur_signature)
        cur_spark_and_logger = spark_and_logger()
        self.spark = cur_spark_and_logger.spark
        self.logger = cur_spark_and_logger.logger
        self.dbg_post_fix = "_DEBUG"
        self.def_df = self.get_def_df()
        self.obs_df = self.get_obs_df()
        if self.is_debug:
            self.cur_target_id = self.get_target_ids_list()[:50]
            self.get_target_ids_list = self.get_target_ids_list_DBG
            self.obs_df = self.get_obs_df_DBG(self.cur_target_id)
            self.action_df = self.get_action_df_DBG(self.cur_target_id)
            self.terminal_df = self.get_terminal_df_DBG(self.cur_target_id)
        else:
            from pyspark.sql.functions import col
            self.cur_target_id = self.get_target_ids_list()
            self.obs_df = self.get_obs_df().where(col("ID").isin(self.get_target_ids_list()))
            self.action_df = self.get_action_df().where(col("ID").isin(self.get_target_ids_list()))
            self.terminal_df = self.get_terminal_df().where(col("ID").isin(self.get_target_ids_list()))


        return

    def get_target_ids_list_DBG(self):
        return self.cur_target_id

    def get_obs_df_DBG(self,target_id):
        if type(self) == data_abstracter:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        '''
        :param target_id: target ID used for debug run
        :return:
        '''
        import pyspark
        #maybe this can be moved to main method?
        try:
            return self.spark.read.parquet(self.cached_obs_df+self.dbg_post_fix)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
        from pyspark.sql.functions import col
        cur_df = self.get_obs_df()
        cur_df.where(col("ID").isin(target_id)).write.save(self.cached_obs_df+self.dbg_post_fix)
        return self.spark.read.parquet(self.cached_obs_df+self.dbg_post_fix)

    def get_action_df_DBG(self,target_id):
        if type(self) == data_abstracter:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        '''

        :param target_id:target ID used for debug run
        :return:
        '''
        import pyspark

        try:
            return self.spark.read.parquet(self.cached_action_df+self.dbg_post_fix)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
        from pyspark.sql.functions import col
        cur_df = self.get_action_df()
        cur_df.where(col("ID").isin(target_id)).write.save(self.cached_action_df+self.dbg_post_fix)
        return self.spark.read.parquet(self.cached_action_df+self.dbg_post_fix)

    def get_terminal_df_DBG(self,target_id):
        if type(self) == data_abstracter:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        '''

        :param target_id:target ID used for debug run
        :return:
        '''
        import pyspark
        try:
            return self.spark.read.parquet(self.cached_terminal_df+self.dbg_post_fix)
        except pyspark.sql.utils.AnalysisException as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.logger.info(message)
            self.logger.info("PROCESS")
        from pyspark.sql.functions import col
        cur_df = self.get_terminal_df()
        cur_df.where(col("ID").isin(target_id)).write.save(self.cached_terminal_df+self.dbg_post_fix)
        return self.spark.read.parquet(self.cached_terminal_df+self.dbg_post_fix)

    @abc.abstractmethod
    def get_obs_df(self):
        pass

    @abc.abstractmethod
    def get_action_df(self):
        '''
        return action dataframe
        :return:
        '''
        pass

    @abc.abstractmethod
    def get_terminal_df(self):
        '''
        return pyspark dataframe with ID, TERMINAL_OUTCOME,SOURCE
        :return:
        '''
        pass

    def get_target_ids_list(self):
        '''
        read file from self.temp_target_id_list_file, if not, create and save them
        :return: list of target ids, MIMIC-IDs with metavision input events and procedures
        '''
        import json
        try:
            f = open(self.temp_target_id_list_file )
            cur_target_id_list = json.loads("".join(f.readlines()))["target_id"]
            f.close()
            return cur_target_id_list
        except IOError as e:
            self.logger.debug("RUN_PROCESS")
        cur_target_id_list =self.get_obs_df().select("ID").distinct().rdd.flatMap(list).collect()
        import os
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
        f = open(self.temp_target_id_list_file,"w")
        f.write(json.dumps({"target_id":cur_target_id_list}))
        f.close()

        return cur_target_id_list

    @abc.abstractmethod
    def get_def_df(self):
        '''
        return definition df: [ITEMID, LABEL, SOURCE]
        :return:
        '''
        pass

    @abc.abstractmethod
    def get_action_itemids(self):
        '''
        return action items after applying inclusion/exclusion criteria
        :return:
        '''
        pass

    def get_target_test_id(self,tr_te_prop = 0.1):
        if type(self) == data_abstracter:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        import json
        try:
            f = open(self.temp_target_test_id_list_file)
            cur_target_test_id_list = json.loads("".join(f.readlines()))["target_id"]
            f.close()
            return cur_target_test_id_list
        except IOError as ex:
            self.logger.debug("RUN_PROCESS")
        cur_list = self.get_target_ids_list()
        from random import shuffle
        shuffle(cur_list)
        target_test_id_list = cur_list[:int(len(cur_list)*tr_te_prop)]

        f = open(self.temp_target_test_id_list_file,"w")
        f.write(json.dumps({"target_id":target_test_id_list}))
        f.close()

        return target_test_id_list

    def get_target_tr_val_id(self,tr_val_prop = 0.2):
        if type(self) == data_abstracter:
            raise NotImplementedError("Method need to be called in sub-class but currently called in base class")

        import json
        try:
            f = open(self.temp_target_tr_val_id_list_file)
            cur_json = json.loads(("".join(f.readlines())).replace("\n",""))
            cur_target_tr_id_list = cur_json["tr_id"]
            cur_target_val_id_list = cur_json["val_id"]
            self.logger.info(cur_json)
            self.logger.info({"TR":cur_target_tr_id_list, "VAL":cur_target_val_id_list})
            f.close()
            return {"TR":cur_target_tr_id_list, "VAL":cur_target_val_id_list}
        except IOError as ex:
            self.logger.debug("RUN_PROCESS")
        cur_all_ids = self.get_target_ids_list()

        cur_test_list = self.get_target_test_id()
        from random import shuffle
        cur_tr_ids = list(set(cur_all_ids).difference(set(cur_test_list)))
        shuffle(cur_tr_ids)
        target_val_ids = cur_tr_ids[:int(len(cur_tr_ids)*tr_val_prop)]
        target_train_ids = cur_tr_ids[int(len(cur_tr_ids)*tr_val_prop):]


        f = open(self.temp_target_tr_val_id_list_file,"w")
        f.write(json.dumps({"tr_id":target_train_ids, "val_id":target_val_ids}))
        f.close()

        return {"TR":target_train_ids, "VAL":target_val_ids}


