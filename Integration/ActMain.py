# -*- coding:utf-8 -*-

# The function of this module is building a process of prediction, include:
#    1. data import and audit;
#    2. data preprocess;
#    3. import model for prediction;
#    4. conduct prediction;
#    5. save prediction result;
#

from BaseModules.FileIO import *
from BaseModules.DataPreProcess import *
from BaseModules.Modeling import *
import h2o


def act_configure(sc, filename):
    """
    
    :return: 
    """
    conf_dict = parse_json2dict_upload(file_name=filename)

    act_conf_dict = dict()
    act_conf_dict['act_data_path'] = conf_dict.get('act_data_path')
    act_conf_dict['learn_fold_path'] = conf_dict.get('learn_fold_path')
    act_conf_dict['act_fold_path'] = conf_dict.get('act_fold_path')
    act_conf_dict['hive_table'] = conf_dict.get('hive_table')

    learn_conf_dict = parse_json2dict_hdfs(sc, act_conf_dict['learn_fold_path'], 'learn_configuration.json')
    act_conf_dict['model_path'] = learn_conf_dict.get('model_path')
    act_conf_dict['algorithm'] = learn_conf_dict.get('train_conf').get('algorithm')
    act_conf_dict['data_sep_symbol'] = learn_conf_dict.get('data_sep_symbol') \
        if learn_conf_dict.get('data_sep_symbol') else ','

    preprocess_info_dict = parse_json2dict_hdfs(sc, act_conf_dict['learn_fold_path'], 'learn_preprocess.json')
    act_conf_dict['type_dict'] = preprocess_info_dict.get('ori_type_dict')

    act_conf_dict['preprocess'] = dict()
    act_conf_dict['preprocess']['remove_vars'] = preprocess_info_dict.get('remove_vars')
    act_conf_dict['preprocess']['fill_dict'] = preprocess_info_dict.get('fill_dict')
    act_conf_dict['preprocess']['id_name'] = preprocess_info_dict.get('id_name')
    act_conf_dict['preprocess']['target_name'] = preprocess_info_dict.get('target_name')

    return act_conf_dict


def act_data_import_audit(spark, act_conf_dict):
    """
    data import and audit
    :return: 
    """
    print '\n--- importing data ---'

    type_dict = act_conf_dict['type_dict']

    if act_conf_dict['hive_table'] is not None:
        #hive_to_hdfs(spark, act_conf_dict['hive_table'], act_conf_dict['act_data_path'])
        data_df = hive_to_hdfs(spark, act_conf_dict['hive_table'])
        data_obj = transfer_sparkdf_as_h2odf(data_df, type_dict)
    else:
        data_obj = import_data_as_frame(data_path=act_conf_dict['act_data_path'],
                                        type_dict=type_dict,
                                        sep=act_conf_dict['data_sep_symbol'],
                                        header=True)

    print '\n--- test id checking ---'
    if act_conf_dict['preprocess']['id_name']:
        print check_id(data_obj['df'], act_conf_dict['preprocess']['id_name'])

    return data_obj


def act_preprocess(data_obj, act_conf_dict):
    """
    data preprocess
    :return: 
    """
    removes = act_conf_dict['preprocess']['remove_vars']
    data_obj['df'] = del_vars(data_df=data_obj['df'], remove_var_lst=removes)

    fill_dict = act_conf_dict['preprocess']['fill_dict']
    data_obj['df'] = fill_missing(data_df=data_obj['df'], fill_dict=fill_dict)

    return data_obj


def bin_100_hist(prediction_df):
    """
    
    :return: 
    """
    prediction_df = prediction_df.sort(by=[2], ascending=False)
    score_list = h2o.as_list(prediction_df[:, [2]], use_pandas=False, header=False)
    score_bins_lst = list()
    for i in range(100):
        up = (i + 1)*0.01
        low = i * 0.01
        # interval = str(low) + '-' + str(up)
        bin_sum = 0
        for j in score_list:
            if low < float(j[0]) <= up:
                bin_sum += 1
        score_bins_lst.append(bin_sum)

    return score_bins_lst


def act_main(spark, config_filename):
    """
    build acting process
    :return: 
    """
    print "SuperAtom 20180125"
    sc = spark.sparkContext

    # parse configuration file
    act_conf_dict = act_configure(sc, filename=config_filename)

    # create learn directory
    make_dir(sc, act_conf_dict['act_fold_path'])

    # import data set as a data frame and conduct  data audit
    data_obj = act_data_import_audit(spark, act_conf_dict)

    # act preprocess
    data_obj = act_preprocess(data_obj, act_conf_dict)

    # model import
    trained_model = load_model(act_conf_dict['model_path'])

    # predict
    prediction_df = prediction(trained_model=trained_model, test_data=data_obj['df'])
    score_bins_dict = dict({'score_bins': bin_100_hist(prediction_df)})

    if act_conf_dict['preprocess']['id_name']:
        prediction_df = data_obj['df'][act_conf_dict['preprocess']['id_name']].cbind(prediction_df)
    prediction_lst = h2o.as_list(prediction_df)

    # export result
    dump_list2csv_hdfs(sc,
                       content_list=prediction_lst,
                       file_path=os.path.join(act_conf_dict['act_fold_path'], 'prediction.csv'))

    dump_dict2json_hdfs(sc,
                        file_path=os.path.join(act_conf_dict['act_fold_path'], 'act_config.json'),
                        dict_content=act_conf_dict)

    dump_dict2json_hdfs(sc,
                        file_path=os.path.join(act_conf_dict['act_fold_path'], 'prediction_bins.json'),
                        dict_content=score_bins_dict)




