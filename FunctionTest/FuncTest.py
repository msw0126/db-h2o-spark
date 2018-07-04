# -*- coding:utf-8 -*-

# The this module is used for functional test:
#

import sys
from pyspark.sql import SparkSession
from BaseModules.DataAudit import *
from DataSetOperation import *
from BaseModules.FileIO import *
from BaseModules.DataPreProcess import *
from BaseModules.Modeling import *
from BaseModules.ClassificationModelInfo import *
from BaseModules.ClusterModelInfo import *
import h2o
from pysparkling import *
import time
import json

os.environ['PYTHON_EGG_CACHE'] = '/tmp/.python-eggs/'
os.environ['PYTHON_EGG_DIR'] = '/tmp/.python-eggs/'


def test_file_io(sc):
    """
    test functions in file io module
    :param: sc: spark context
    :return: nothing(on log)
    """
    timestamp_str = str(int(time.time()))

    test_data_path = 'hdfs://172.31.22.94:8020/home/hadoop/h2o-spark/test-data/test_small_data.csv'
    test_data_dict_path = 'hdfs://172.31.22.94:8020/home/hadoop/h2o-spark/test-data/test_small_data_dict.csv'
    test_json_dir = 'hdfs://172.31.22.94:8020/home/hadoop/h2o-spark/test-data'
    test_json_filename = 'test_json.json'
    upload_json_name = 'test_upload_json.json'
    upload_csv_name = 'test_small_data_dict.csv'

    test_new_dir = 'hdfs://172.31.22.94:8020/home/hadoop/h2o-spark/test-data/test_file_' + timestamp_str

    print '\n--- test make directory ---'
    make_dir(sc, test_new_dir)

    print '\n--- test parse upload json ---'
    parsed_upload_json = parse_json2dict_upload(upload_json_name)
    print json.dumps(parsed_upload_json, indent=1)

    print '\n--- test parse hdfs json ---'
    parsed_hdfs_json = parse_json2dict_hdfs(sc, test_json_dir, test_json_filename)
    print json.dumps(parsed_hdfs_json, indent=1)

    print '\n--- test parse upload csv ---'
    parsed_upload_csv = parse_csv2list_upload(upload_csv_name)
    print parsed_upload_csv

    print '\n--- test parse hdfs csv ---'
    parsed_hdfs_csv = parse_csv2list_hdfs(test_data_dict_path)
    print parsed_hdfs_csv

    print '\n--- test dump dict to json ---'
    dump_dict2json_hdfs(sc, test_new_dir + '/dumped_json.json', parsed_upload_json)

    print '\n--- test dump list to csv ---'
    dump_list2csv_hdfs(sc, parsed_upload_csv, test_new_dir + '/dumped_csv.csv')

    print '\n--- test import data dict upload ---'
    upload_dict, col_list = import_data_dict(upload_csv_name,
                                             numeric_types=['int', 'double'], factor_types=['string', 'char'])
    print json.dumps(upload_dict, indent=1)
    print col_list

    print '\n--- test import data dict hdfs ---'
    hdfs_dict, col_list = import_data_dict(test_data_dict_path, upload_dict)
    print json.dumps(hdfs_dict, indent=1)
    print col_list

    print '\n--- test import data to data frame ---'
    data_obj = import_data_as_frame(test_data_path, upload_dict)
    print data_obj['df']
    data_obj['df'].summary()
    print data_obj['schema']
    print data_obj['info']


def test_dataset_operation(data_obj):
    """
    test functions in data set operation module
    :param data_obj: a data object with its frame and dict and other info
    :return: nothing(on log)
    """
    print '\n--- test data split ---'
    test_df_1, test_df_2 = split_data(data_obj['df'], 0.3, stratified=True, stratified_col='label')
    print test_df_1
    print test_df_2

    print '\n--- test data merge ---'
    test_df_3 = merge_data(df_1=test_df_2, df_2=test_df_1, left_all=True, right_all=False, df_1_keys=['label'],
                           df_2_keys=['label'])
    print test_df_3


def test_data_audit(data_obj, id_name, target_name):
    """
    test functions in data audit module
    note that type dict audit has been done while import dict file
    :param data_obj: a data object with its frame and dict and other info
    :param id_name: id name
    :param target_name: target name
    :return: nothing(on log)
    """
    print '\n--- test id checking ---'
    print check_id(data_obj['df'], id_name)

    print '\n--- test label checking ---'
    print check_label(data_obj['df'], target_name)

    print '\n--- test data volume checking ---'
    print check_data_vol(data_obj['df'], row_low_limit=5)


def test_preprocess(data_obj, id_name, target_name):
    """
    test functions in preprocess module
    :param data_obj: a data frame
    :param id_name: id name
    :param target_name: label name
    :return: nothing(on log)
    """
    test_data_df = data_obj['df']
    test_data_dict = data_obj['schema']
    numeric_vars = [i for i in test_data_dict.keys() if test_data_dict[i] == 'numeric']
    factor_vars = [i for i in test_data_dict.keys() if test_data_dict[i] == 'factor']
    factor_vars.remove(id_name)
    factor_vars.remove(target_name)
    remove_indexs = [0, 1, 3]
    remove_vars = test_data_df.names[:3]
    sample_ratio = 0.3
    sample_num = 5
    fill_dict = {'V8': 10, 'V9': 'C110'}

    print '\n--- test calculate factor variable statistics ---'
    print process_id_label_type(test_data_df, id_name, target_name)

    print '\n--- test calculate factor variable statistics ---'
    factor_statics_dict = cal_factor_stat(test_data_df, factor_vars)
    print json.dumps(factor_statics_dict, indent=1)

    print '\n--- test calculate numeric variable statistics ---'
    numeric_statics_dict = cal_numeric_stat(test_data_df, numeric_vars)
    print json.dumps(numeric_statics_dict, indent=1)

    print '\n--- test calculate missing variable number of each sample ---'
    row_miss_lst, id_miss_map = cal_samples_miss(test_data_df, id_name)
    print row_miss_lst
    print id_miss_map

    print '\n--- test calculate level numbers of factor variables ---'
    levels_dict = cal_vars_levels(test_data_df)
    print json.dumps(levels_dict, indent=1)

    print '\n--- test calculate missing sample number of each variable ---'
    col_miss_dict = cal_vars_miss(test_data_df)
    print json.dumps(col_miss_dict, indent=1)

    print '\n--- test calculate standard deviation of numeric variable ---'
    col_std_dict = cal_vars_std(test_data_df)
    print json.dumps(col_std_dict, indent=1)

    print '\n--- test delete samples ---'
    print del_samples(test_data_df, remove_index_lst=remove_indexs)

    print '\n--- test delete variables ---'
    print del_vars(test_data_df, remove_var_lst=remove_vars)

    print '\n--- test calculate quantity of each factor in label  ---'
    count_dict = label_static(test_data_df[target_name])
    print json.dumps(count_dict, indent=1)

    print '\n--- test random sampling in proportion ---'
    print 'original row number: ' + str(test_data_df.shape[0])
    sampled_df = random_sampling(test_data_df, ratio=sample_ratio)
    print 'sampled row number in proportion: ' + str(sampled_df.shape[0])

    print '\n--- test random sampling by sub number ---'
    print 'original row number: ' + str(test_data_df.shape[0])
    sampled_df = random_sampling(test_data_df, sub_num=sample_num)
    print 'sampled row number in proportion: ' + str(sampled_df.shape[0])

    print '\n--- test fill missing values ---'
    filled_df = fill_missing(test_data_df, fill_dict)
    print filled_df

    print '\n--- test cut numeric variable into bins by quantiles ---'
    bin_df, numeric_bin_dict = numeric_quantile_bin(test_data_df, numeric_vars)
    print bin_df
    print json.dumps(numeric_bin_dict, indent=1)

    print '\n--- test calculate woe ---'
    woe_dict = cal_woe(bin_df, numeric_vars + factor_vars, target_name)
    print json.dumps(woe_dict, indent=1)

    print '\n--- test order factor levels by woe ---'
    woe_factor_dict = dict()
    for i in factor_vars:
        woe_factor_dict[i] = woe_dict[i]
    factor_level_dict = factor_woe_order(woe_factor_dict)
    print json.dumps(factor_level_dict, indent=1)

    print '\n--- test merge bins by chi2 test for woe ---'
    bin_dict = dict(numeric_bin_dict.items() + factor_level_dict.items())
    woe_dict, bin_dict = bins_merge(woe_dict, bin_dict)
    print json.dumps(woe_dict, indent=1)
    print json.dumps(bin_dict, indent=1)

    print '\n--- test calculate iv ---'
    iv_dict = cal_iv(woe_dict)
    print json.dumps(iv_dict, indent=1)


def test_classification_model_properties(trained_model, algo, xval=True):
    """
    
    :param trained_model: 
    :param algo:
    :param xval:
    :return: 
    """
    print '\n----- actual params -----'
    actual_params_dict = get_actual_params(trained_model)
    print actual_params_dict

    print '\n----- model summary -----'
    if algo != 'Stacking':
        summary_df, summary_lst = model_summary(trained_model)
        print summary_df
        print summary_lst

    if xval:
        print '\n----- cross validation summary -----'
        cv_summary_df, cv_summary_lst = cross_validation_summary(trained_model)
        print cv_summary_df
        print cv_summary_lst

    print '\n----- variable importance -----'
    if algo in ['DL', 'GBM', 'RF']:
        varimp_df, varimp_lst = var_important(trained_model)
        print varimp_df
        print varimp_lst

    print '\n----- linear coefficients -----'
    if algo == 'LR':
        coef_df, coef_lst = linear_coef(trained_model)
        print coef_df
        print coef_lst


def test_classification_model_metric(metric_obj):
    """
    
    :param metric_obj: 
    :return: 
    """
    print '\n----- synthetical metrics -----'
    metric_dict = synthetical_metrics(metric_obj)
    print metric_dict

    print '\n----- metrics by threshold-----'
    metric_dict = metrics_by_threshold(metric_obj)
    print metric_dict

    print '\n----- thresholds by max metric scores-----'
    metric_dict = max_metric_threshold(metric_obj)
    print metric_dict

    print '\n----- gains / lift-----'
    gains_lift_df, gains_lift_lst = gains_lift(metric_obj)
    print gains_lift_df
    print gains_lift_lst

    print '\n----- max criteria metric scores thresholds table-----'
    max_criteria_metric_df, max_criteria_metric_lst = max_criteria_metric(metric_obj)
    print max_criteria_metric_df
    print max_criteria_metric_lst

    print '\n----- confusion matrix-----'
    c_matrix_df, c_matrix_lst = confusion_matrix(metric_obj)
    print c_matrix_df
    print c_matrix_lst

    print '\n----- ks value-----'
    thresholds_scores_df_s, ks_val, ks_threshold = cal_ks_val(metric_obj)
    print thresholds_scores_df_s
    print ks_val
    print ks_threshold


def test_modeling(df, id_name, target_name):
    """
    test functions in preprocess module
    :param df: a data frame
    :param id_name: id name
    :param target_name: label name
    :return: nothing(on log)
    """
    import datetime
    df[target_name] = df[target_name].asfactor()
    x = df.names
    x.remove(id_name)
    x.remove(target_name)
    y = target_name
    sample_size = df.shape[0]
    train_df, test_df = df.split_frame(ratios=[0.7])
    df_1, df_2 = train_df.split_frame(ratios=[0.7])

    for algo in ['DL', 'LR', 'GBM', 'NB', 'RF']:

        print '\n===== train %s algorithm with kfold =====' % algo
        start_time = datetime.datetime.now()

        estimator, hparams = get_algorithm_estimator(algo, sample_size=sample_size, xval=True, nfolds=3)
        gs_model = grid_search(estimator, hparams)
        trained_gs_model = training(gs_model, x=x, y=y, train_data=train_df)
        best_model_1 = get_gridsearch_best(trained_gs_model, metric='auc')

        end_time = datetime.datetime.now()
        print 'running time: ' + str((end_time - start_time).seconds)
        print best_model_1

        print '\n===== export model properties ====='
        test_classification_model_properties(best_model_1, algo, xval=True)

        print '\n===== export model metrics ====='
        metric_obj = get_model_metric_obj(best_model_1, xval=True)
        test_classification_model_metric(metric_obj)

        print '\n===== predict ====='
        predict = prediction(best_model_1, test_df)
        print predict

        print '\n===== export test metric scores====='
        test_metric_obj = test_perfomance(best_model_1, test_df)
        test_classification_model_metric(test_metric_obj)

        print '++++++++++++++++++++++++++++++++++++++++++++'

        print '\n--- test %s algorithm without kfold cv ---' % algo

        start_time = datetime.datetime.now()

        estimator, hparams = get_algorithm_estimator(algo, sample_size=df_1.shape[0])
        gs_model = grid_search(estimator, hparams)
        trained_gs_model = training(gs_model, x=x, y=y, train_data=df_1, valid_data=df_2)
        best_model_2 = get_gridsearch_best(trained_gs_model, metric='auc')

        end_time = datetime.datetime.now()
        print 'running time: ' + str((end_time - start_time).seconds)
        print best_model_2

        print '\n===== export model properties ====='
        test_classification_model_properties(best_model_2, algo, xval=False)

        print '\n===== export model metrics ====='
        metric_obj = get_model_metric_obj(best_model_2, valid=True)
        test_classification_model_metric(metric_obj)

        print '\n===== predict ====='
        predict = prediction(best_model_2, test_df)
        print predict

        print '\n===== export test metric scores====='
        test_metric_obj = test_perfomance(best_model_2, test_df)
        test_classification_model_metric(test_metric_obj)

        print '++++++++++++++++++++++++++++++++++++++++++++'

    # Stacking

    stacking_test=True

    if stacking_test:

        print '\n--- test Stacking with RF and LR ---'

        estimator_1, hparams_1 = get_algorithm_estimator(algo_aka='RF', sample_size=sample_size,
                                                         xval=True, nfolds=3, for_stacking=True)
        trained_model_1 = training(model=estimator_1, x=x, y=y, train_data=df_1)

        estimator_2, hparams_2 = get_algorithm_estimator(algo_aka='LR', sample_size=sample_size,
                                                         xval=True, for_stacking=True)
        gs_model = grid_search(estimator_2, hparams_2)
        trained_gs_model = training(gs_model, x=x, y=y, train_data=df_1)
        trained_model_2 = get_gridsearch_best(trained_gs_model, metric='auc')

        stacking_estimator = get_algorithm_estimator(algo_aka='Stacking',
                                                     model_lst=[trained_model_1.model_id, trained_model_2.model_id])
        trained_model = training(stacking_estimator, x=x, y=y, train_data=df_1, valid_data=df_2)
        print trained_model

        print '\n===== export model properties ====='
        test_classification_model_properties(trained_model, 'Stacking', xval=False)

        print '\n===== export model metrics ====='
        metric_obj = get_model_metric_obj(trained_model, valid=True)
        test_classification_model_metric(metric_obj)

        print '\n===== predict ====='
        predict = prediction(trained_model, test_df)
        print predict

        print '\n===== export test metric scores====='
        test_metric_obj = test_perfomance(trained_model, test_df)
        test_classification_model_metric(test_metric_obj)

        print '++++++++++++++++++++++++++++++++++++++++++++'

    # K-means
    kmeans_test = True

    if kmeans_test:
        print '\n--- test k means ---'

        estimator, hparams = get_algorithm_estimator(algo_aka='KM', sample_size=sample_size, xval=True, nfolds=3)
        gs_model = grid_search(estimator, hparams)
        trained_gs_model = training(gs_model, x=(x + [y]), train_data=train_df)
        trained_model = get_gridsearch_best(trained_gs_model, metric='betweenss', decreasing=True)
        print trained_model

        print '\n===== export model properties ====='

        print '\n--- k means model properties ---'
        result_dict = get_km_model_properties(trained_model)
        print result_dict

        print '\n--- k means model properties on all sets ---'
        result_dict = get_km_model_properties_by_set(trained_model)
        print result_dict

        print '\n===== predict ====='
        predict = prediction(trained_model, test_df)
        print predict

        print '\n===== export test metric scores====='
        test_metric_obj = test_perfomance(trained_model, test_df)
        metrics_dict = get_km_test_metrics(test_metric_obj)
        print metrics_dict
        print metrics_dict['betweenss']

        print '++++++++++++++++++++++++++++++++++++++++++++'

if __name__ == '__main__':

    spark = SparkSession.builder.enableHiveSupport().appName("h2o-Atom-Test").getOrCreate()
    sc = spark.sparkContext
    hc = H2OContext.getOrCreate(spark)

    # ---test file io
    # test_file_io(sc)

    # ---test data audit

    # test_data_path = 'hdfs://172.31.22.94:8020/home/hadoop/h2o-spark/test-data/test_small_data.csv'
    # upload_csv_name = 'test_small_data_dict.csv'
    # id_name, target_name = 'id', 'label'
    # upload_dict, col_list = import_data_dict(upload_csv_name,
    #                                          numeric_types=['int', 'double'], factor_types=['string', 'char'])
    # data_obj = import_data_as_frame(test_data_path, upload_dict)
    # test_data_audit(data_obj, id_name=id_name, target_name=target_name)

    # ---test data set operation
    # test_dataset_operation(data_obj)

    # ---test preprocess
    # test_preprocess(data_obj, id_name, target_name)

    # ---test modeling
    df = h2o.import_file('hdfs://172.31.22.94:8020/home/hadoop/h2o-spark/test-data/prostate.csv')

    id_name = 'ID'
    target_name = 'CAPSULE'
    df[target_name] = df[target_name].asfactor()
    test_modeling(df, id_name, target_name)
