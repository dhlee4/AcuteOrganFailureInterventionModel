'''
Running OK so far.
'''


#responsibility: 1) Generalize the given dataset,2) filter bad quality data entry, 3) flatten them with NULL. 
#Current bad quality filtering out criteria: continuous variable - IQR filter, categorical variable - less frequent categorical entry

#spark.sparkContext.setCheckpointDir("/gscratch/stf/dhlee4/spark_checkpoint/")

#V1 : DNR, DNI, CMO Excluded. Metavision only, categorical variable deciding factor: cardinality is less than 20, variables with majority numeric values and some string treated as numeric variables, and string values are treated separately as categorical variable during feature representation -- rewrite this.

from data_preprocessor import data_preprocessor
from mimic_data_abstracter import mimic_data_abstracter

class mimic_preprocessor(mimic_data_abstracter,data_preprocessor):
    def __init__(self,target_env=None, is_debug=True,cur_signature = ""):
        from datetime import datetime
        mimic_data_abstracter.__init__(self,target_env,is_debug,cur_signature=cur_signature)
        data_preprocessor.__init__(self)
        self.postfix = "V1.0_B{0}_{1}".format(cur_signature, "D" if self.is_debug else "")
        self.out_file_name = self.home_dir + "/final_processed_out_{0}".format(self.postfix)
        self.voca_name=self.home_dir+"/voca_list_{0}".format(self.postfix)
        if self.is_debug:
            self.th_range = [0.3]
        else:
            self.th_range = [0.5]
            # TODO Something wrong in the logger. Need to figure this out
            #self.logger.debug( self.th_range)





if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_env", help="target environment, default=Test")
    args = parser.parse_args()
    if args.target_env:
        cur_target_env = args.target_env
    else:
        cur_target_env = None

    cur_preprocessor = mimic_preprocessor(target_env=cur_target_env,is_debug=True)
    cur_preprocessor.run_preprocessor()
