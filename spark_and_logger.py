
#Just for Mac Pro
import logging
#TODO: Need to make test environment for both python 2 and python 3
import sys
cur_python_ver = sys.version_info
is_python3=None
if cur_python_ver[0] == 3:
    is_python3=True
else:
    is_python3=False


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

class spark_and_logger(object):
    __shared_state={}

    def __init__(self,std_out_log = False):
        self.__dict__ = self.__shared_state
        self.std_out_log = std_out_log
        self.state = 'Init'

    def __str__(self):
        return self.state

    def __getattr__(self,attr):
        if attr == "spark":
            if not is_python3:
                if not self.__shared_state.has_key("spark"):
                    from pyspark.sql import SparkSession
                    # TODO allow users to specify the configuration
                    from pyspark import SparkConf
                    self.__shared_state["spark"]= SparkSession.builder\
                        .config([("spark.driver.memory","12g")
                              , ("spark.executor.memory", "12g")
                              , ('spark.driver.maxResultsSize','0')
                              , ('spark.speculation','true')])\
                        .getOrCreate()
                    #if self.home_dir:
                        #self.__shared_state["spark"].sparkContext.setCheckpointDir(self.home_dir + "tmp_chkpoint")
                    #else:
                        #raise Exception("home_dir not setted for checkpoint")
                return self.__shared_state["spark"]
            else:
                if "spark" not in self.__shared_state:
                    from pyspark.sql import SparkSession
                    self.__shared_state["spark"]= SparkSession.builder.getOrCreate()
                    if self.home_dir:
                        self.__shared_state["spark"].sparkContext.setCheckpointDir(self.home_dir + "tmp_chkpoint")
                    else:
                        raise Exception("home_dir not setted for checkpoint")
                return self.__shared_state["spark"]
        elif attr == "logger":
            if not is_python3:
                if not self.__shared_state.has_key("logger"):
                    import logging
                    '''logger = logging.getLogger()
                    FORMAT = "[%(asctime)s%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
                    '''
                    import os
                    logging.basicConfig(level=logging.INFO,
                                        format="[%(asctime)s%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
                                        filename="{0}/{1}.log".format("./", "{0}_logger".format(os.environ["PHD_WORK_ENVIRONMENT"])))

                    import sys
                    if self.std_out_log:
                        stdout_logger = logging.getLogger('STDOUT')
                        sl = StreamToLogger(stdout_logger, logging.INFO)
                        sys.stdout = sl

                        stderr_logger = logging.getLogger('STDERR')
                        sl = StreamToLogger(stderr_logger, logging.ERROR)
                        sys.stderr = sl

                    logging.getLogger().info("LOGGING STARTS")

                    self.__shared_state["logger"] = logging.getLogger()
                else:
                    import logging
                    '''logger = logging.getLogger()
                    FORMAT = "[%(asctime)s%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
                    '''
                    import os
                    logging.basicConfig(level=logging.INFO,
                                        format="[%(asctime)s%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
                                        filename="{0}/{1}.log".format("./", "{0}_logger".format(
                                            os.environ["PHD_WORK_ENVIRONMENT"])))

                    import sys
                    if self.std_out_log:
                        stdout_logger = logging.getLogger('STDOUT')
                        sl = StreamToLogger(stdout_logger, logging.INFO)
                        sys.stdout = sl

                        stderr_logger = logging.getLogger('STDERR')
                        sl = StreamToLogger(stderr_logger, logging.ERROR)
                        sys.stderr = sl

                    logging.getLogger().info("LOGGING STARTS")

                    self.__shared_state["logger"] = logging.getLogger()
                return self.__shared_state["logger"]
        elif attr == "home_dir":
            if not is_python3:
                if not self.__shared_state.has_key("home_dir"):
                    raise Exception("home_dir not set")
                else:
                    return self.__shared_state["home_dir"]
            else:
                if "home_dir" not in self.__shared_state:
                    raise Exception("home_dir not set")
                else:
                    return self.__shared_state["home_dir"]
        else:
            raise Exception("Is there any other variable needed?{0}".format(attr))

if __name__=="__main__":
    cur_class = spark_and_logger()
    cur_class.qq = "AA"
    other_class = spark_and_logger()
