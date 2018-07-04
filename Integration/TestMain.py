# -*- coding:utf-8 -*-

# The function of this module is building a process of test, include:
#    1. data import and audit;
#    2. data preprocess;
#    3. import model for test;
#    4. conduct test;
#    5. save test result;
#

from BaseModules.FileIO import *
from BaseModules.DataPreProcess import *
from BaseModules.Modeling import *
from BaseModules.Report import *
import h2o


def test_configure(sc, filename):
    """

    :return: 
    """
    conf_dict = parse_json2dict_upload(file_name=filename)

    test_conf_dict = dict()
    test_conf_dict['test_data_path'] = conf_dict.get('test_data_path')
    test_conf_dict['act_fold_path'] = conf_dict.get('act_fold_path')
    test_conf_dict['learn_fold_path'] = conf_dict.get('learn_fold_path')
    test_conf_dict['test_fold_path'] = conf_dict.get('test_fold_path')
    test_conf_dict['hive_table'] = conf_dict.get('hive_table')

    if isinstance(test_conf_dict['act_fold_path'], str) and len(test_conf_dict['act_fold_path']) > 0:
        act_conf_dict = parse_json2dict_hdfs(sc, test_conf_dict['act_fold_path'], 'act_config.json')
        test_conf_dict['algorithm'] = act_conf_dict.get('algorithm')
        test_conf_dict['type_dict'] = act_conf_dict.get('type_dict')
        test_conf_dict['model_path'] = act_conf_dict.get('model_path')
        test_conf_dict['data_sep_symbol'] = act_conf_dict.get('data_sep_symbol') \
            if act_conf_dict.get('data_sep_symbol') else ','

        test_conf_dict['preprocess'] = dict()
        test_conf_dict['preprocess']['remove_vars'] = act_conf_dict.get('preprocess').get('remove_vars')
        test_conf_dict['preprocess']['fill_dict'] = act_conf_dict.get('preprocess').get('fill_dict')
        test_conf_dict['preprocess']['id_name'] = act_conf_dict.get('preprocess').get('id_name')
        test_conf_dict['preprocess']['target_name'] = act_conf_dict.get('preprocess').get('target_name')

    if isinstance(test_conf_dict['learn_fold_path'], str) and len(test_conf_dict['learn_fold_path']) > 0:
        learn_conf_dict = parse_json2dict_hdfs(sc, test_conf_dict['learn_fold_path'], 'learn_configuration.json')

        test_conf_dict['algorithm'] = learn_conf_dict.get('train_conf').get('algorithm')
        test_conf_dict['model_path'] = learn_conf_dict.get('model_path')
        test_conf_dict['data_sep_symbol'] = learn_conf_dict.get('data_sep_symbol') \
            if learn_conf_dict.get('data_sep_symbol') else ','

        preprocess_info_dict = parse_json2dict_hdfs(sc, test_conf_dict['learn_fold_path'], 'learn_preprocess.json')
        test_conf_dict['type_dict'] = preprocess_info_dict.get('ori_type_dict')

        test_conf_dict['preprocess'] = dict()
        test_conf_dict['preprocess']['remove_vars'] = preprocess_info_dict.get('remove_vars')
        test_conf_dict['preprocess']['fill_dict'] = preprocess_info_dict.get('fill_dict')
        test_conf_dict['preprocess']['id_name'] = preprocess_info_dict.get('id_name')
        test_conf_dict['preprocess']['target_name'] = preprocess_info_dict.get('target_name')

    return test_conf_dict


def test_data_import_audit(spark, test_conf_dict):
    """
    data import and audit
    :return: 
    """
    print '\n--- importing data ---'

    type_dict = test_conf_dict['type_dict']

    if test_conf_dict['hive_table'] is not None:
        # hive_to_hdfs(spark, test_conf_dict['hive_table'], test_conf_dict['test_data_path'])
        data_df = hive_to_hdfs(spark, test_conf_dict['hive_table'])
        data_obj = transfer_sparkdf_as_h2odf(data_df, type_dict)
    else:
        data_obj = import_data_as_frame(data_path=test_conf_dict['test_data_path'],
                                        type_dict=type_dict,
                                        sep=',',
                                        header=True)

    print '\n--- test id checking ---'
    if test_conf_dict['preprocess']['id_name']:
        print check_id(data_obj['df'], test_conf_dict['preprocess']['id_name'])

    print '\n--- test label checking ---'
    if test_conf_dict['preprocess']['target_name']:
        print check_label(data_obj['df'], test_conf_dict['preprocess']['target_name'])

    return data_obj


def test_preprocess(data_obj, test_conf_dict):
    """
    data preprocess
    :return: 
    """
    removes = test_conf_dict['preprocess']['remove_vars']
    data_obj['df'] = del_vars(data_df=data_obj['df'], remove_var_lst=removes)

    fill_dict = test_conf_dict['preprocess']['fill_dict']
    data_obj['df'] = fill_missing(data_df=data_obj['df'], fill_dict=fill_dict)

    return data_obj


def test_metrics_save(sc, algo, metrics_obj, test_fold_path):
    """

    :param metric_obj: 
    :return: 
    """
    if algo == 'KM':
        model_metrics_dict, model_metrics_str \
            = export_kmeans_model_metrics_report(metrics_obj=metrics_obj)
    elif algo in ['DL', 'LR', 'GBM', 'NB', 'RF', 'Stacking']:
        model_metrics_dict, model_metrics_str, gains_lift_lst, thresholds_metrics_100_lst \
            = export_classification_model_metrics_report(metrics_obj=metrics_obj)

        dump_list2csv_hdfs(sc, gains_lift_lst, os.path.join(test_fold_path, 'gains-lift-table.csv'))
        dump_list2csv_hdfs(sc, thresholds_metrics_100_lst,
                           os.path.join(test_fold_path, 'metrics-on-100-thresholds.csv'))

    else:
        model_metrics_dict, model_metrics_str = dict(), ''

    dump_dict2json_hdfs(sc,
                        file_path=os.path.join(test_fold_path, 'test_metrics.json'),
                        dict_content=model_metrics_dict)

    write_to_hdfs(sc,
                  file_path=os.path.join(test_fold_path, 'test_metrics.txt'),
                  content=model_metrics_str)


def test_main(spark, config_filename):
    """
    build test process
    :return: 
    """
    print "SuperAtom 20180125"
    sc = spark.sparkContext

    # parse configuration file
    test_conf_dict = test_configure(sc, filename=config_filename)

    # create learn directory
    make_dir(sc, test_conf_dict['test_fold_path'])

    # import data set as a data frame and conduct  data audit
    data_obj = test_data_import_audit(spark, test_conf_dict)

    # act preprocess
    data_obj = test_preprocess(data_obj, test_conf_dict)

    # model import
    trained_model = h2o.load_model(test_conf_dict['model_path'])

    # predict
    metrics_obj = test_perfomance(trained_model=trained_model, test_data=data_obj['df'])

    # export result
    test_metrics_save(sc, algo=test_conf_dict['algorithm'],
                      metrics_obj=metrics_obj, test_fold_path=test_conf_dict['test_fold_path'])

