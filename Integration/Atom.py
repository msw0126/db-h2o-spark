# -*- coding:utf-8 -*-
import sys
import os
import h2o
from pysparkling import *
from pyspark.sql import SparkSession
from Integration.LearnMain import learn_main
from Integration.ActMain import act_main
from Integration.TestMain import test_main
from FileIO import *

os.environ['PYTHON_EGG_CACHE'] = '/tmp/.python-eggs/'
os.environ['PYTHON_EGG_DIR'] = '/tmp/.python-eggs/'

def test(spark):
    """
    
    :param spark: 
    :return: 
    """
    sc = spark.sparkContext

    train_df = h2o.import_file(path='hdfs://node1:8020//taoshu/atom/German/german_credit.csv', header=1, sep=',')

    y = 'V21'
    x = train_df.names
    x.remove(y)
    x.remove('id')
    from h2o.estimators.xgboost import H2OXGBoostEstimator
    xgb = H2OXGBoostEstimator()
    trained_model = xgb.train(x=x, y='V21', training_frame=train_df)
    print trained_model
    model_saved_path = h2o.save_model(trained_model, path='hdfs://node1:8020/taoshu/atom')
    model = h2o.load_model(model_saved_path)
    prediction = model.predict(trained_model)
    print prediction


if __name__ == '__main__':

    spark = SparkSession\
        .builder\
        .enableHiveSupport()\
        .appName("h2o-Atom-Test")\
        .getOrCreate()

    hc = H2OContext.getOrCreate(spark)

    argv = sys.argv[1]

    # test(spark)
    # argv = 1

    if argv == 'Learn':

        config_filename = 'learn_config.json'
        learn_main(spark, config_filename)
        spark.stop()

    elif argv == 'Act':

        config_filename = 'act_config.json'
        act_main(spark, config_filename)
        spark.stop()

    elif argv == 'Test':

        config_filename = 'test_config.json'
        test_main(spark, config_filename)
        spark.stop()
