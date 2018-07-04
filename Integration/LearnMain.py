# -*- coding:utf-8 -*-

# The function of this module is building a process of learning the best model, include:
#    1. data import and audit;
#    2. data preprocess;
#    3. algorithm tuning;
#    4. train a model on entire data with the best parameters;
#    5. save the best model and learning report;
#

import copy
import os
from BaseModules.FileIO import *
from BaseModules.Modeling import *
from BaseModules.Report import *
from Integration.LearnPreprocess import *
from Integration.LearnTrain import *


def learn_configure(conf_filename):
    """
    
    :param conf_filename: 
    :return: 
    """
    print '\n--- parse configuration ---'
    conf_dict = parse_json2dict_upload(conf_filename)

    dict_path = conf_dict.get('dict_path')
    data_types = conf_dict.get('data_types')
    if data_types:
        numeric_types = data_types.get('numeric')
        factor_type = data_types.get('factor')
    else:
        numeric_types = None
        factor_type = None
    type_dict, col_name_lst = import_data_dict(dict_path,
                                               numeric_types=numeric_types,
                                               factor_types=factor_type)
    print type_dict
    print col_name_lst
    learn_conf_dict = dict()
    learn_conf_dict['train_data_path'] = conf_dict.get('train_data_path')
    learn_conf_dict['learn_fold_path'] = conf_dict.get('learn_fold_path')
    learn_conf_dict['hive_table'] = conf_dict.get('hive_table')
    learn_conf_dict['data_sep_symbol'] = ',' if conf_dict.get('data_sep_symbol') is None else conf_dict.get('data_sep_symbol')
    learn_conf_dict['data_missing_symbol'] = ["null", "Null", "NULL", "NaN", "nan", "Na", "NA", "N/A", "None", "NONE", "\\N", "", "?"] if conf_dict.get('data_missing_symbol') is None else conf_dict.get('data_missing_symbol')
    learn_conf_dict['data_types'] = conf_dict.get('data_types')

    learn_conf_dict['id_name'] = conf_dict.get('id_name')
    learn_conf_dict['target_name'] = conf_dict.get('target_name')

    # preprocess configuration
    cal_statics_universal = conf_dict.get('cal_statics_universal')
    cal_statics_sampling = conf_dict.get('cal_statics_sampling')
    max_factor_prop = conf_dict.get('max_factor_prop')
    max_variable_miss_prop = conf_dict.get('max_variable_miss_prop')
    max_sample_miss_prop = conf_dict.get('max_sample_miss_prop')
    sampling_method = conf_dict.get('sampling_method')
    unbalanced_cutoff = conf_dict.get('unbalanced_cutoff')
    fill_dict = conf_dict.get('fill_dict')
    calculate_iv = conf_dict.get('cal_iv')
    max_levels_amount = 200
    learn_conf_dict['preprocess_conf'] = dict({'max_factor_levels_prop': max_factor_prop,
                                               'max_variable_miss_prop': max_variable_miss_prop,
                                               'max_sample_miss_prop': max_sample_miss_prop,
                                               'sampling_method': sampling_method,
                                               'unbalanced_cutoff': unbalanced_cutoff,
                                               'cal_statics_universal': cal_statics_universal,
                                               'cal_statics_sampling': cal_statics_sampling,
                                               'fill_dict': fill_dict,
                                               'cal_iv': calculate_iv,
                                               'max_levels_amount': max_levels_amount})

    print learn_conf_dict['preprocess_conf']

    # training configuration
    algo = conf_dict.get('algorithm')
    hparams = conf_dict.get('hparams') if isinstance(conf_dict.get('hparams'), dict) else None
    cv_k = conf_dict.get('cv_k') if isinstance(conf_dict.get('cv_k'), int) else None
    # cv_k = 0,1 will do 7/3, train/validation

    if algo == 'KM':
        learn_conf_dict['preprocess_conf']['cal_iv'] = False
        learn_conf_dict['target_name'] = None

    learn_conf_dict['train_conf'] = dict({'algorithm': algo, 'hparams': hparams, 'cv_k': cv_k})
    print learn_conf_dict['train_conf']

    return learn_conf_dict, type_dict, col_name_lst


def learn_data_import_audit(spark, learn_conf_dict, type_dict):
    """
    data import and audit
    :return: 
    """
    print '\n--- importing data ---'
    if learn_conf_dict['hive_table'] is not None:
        # hive_to_hdfs(spark, learn_conf_dict['hive_table'], learn_conf_dict['train_data_path'])
        data_df = hive_to_hdfs(spark, learn_conf_dict['hive_table'])
        data_obj = transfer_sparkdf_as_h2odf(data_df, type_dict)
    else:
        data_obj = import_data_as_frame(data_path=learn_conf_dict['train_data_path'],
                                        type_dict=type_dict,
                                        sep=learn_conf_dict['data_sep_symbol'],
                                        na_lst=learn_conf_dict['data_missing_symbol'])

    print '---- test debug learn_data_import_audit 1 ---'
    print data_obj['df'].shape

    print '\n--- test id checking ---'
    if learn_conf_dict['id_name']:
        print check_id(data_obj['df'], learn_conf_dict['id_name'])

    print '\n--- test label checking ---'
    if learn_conf_dict['target_name']:
        print check_label(data_obj['df'], learn_conf_dict['target_name'])

    print '\n--- test data volume checking ---'
    print check_data_vol(data_obj['df'],
                         id_name=learn_conf_dict['id_name'],
                         target_name=learn_conf_dict['target_name'],
                         row_low_limit=1)

    return data_obj


def learn_preprocess(data_obj, learn_conf_dict):
    """
    data preprocess
    :return: 
    """

    data_obj['info']['ori_type_dict'] = copy.deepcopy(data_obj['schema'])
    id_name = learn_conf_dict['id_name']
    target_name = learn_conf_dict['target_name']
    report_str = ''
    report_str += 'id name: ' + str(id_name) + '\n'
    report_str += 'target name: ' + str(target_name) + '\n\n'

    ori_nrows = data_obj['df'].shape[0]
    ori_ncols = data_obj['df'].shape[1]
    report_str += 'original sample amount: ' + str(ori_nrows) + '\n'
    report_str += 'original variable amount: ' + str(ori_ncols) + '\n\n'

    variables = data_obj['df'].names
    if id_name:
        variables.remove(id_name)
    if target_name:
        variables.remove(target_name)

    numeric_vars = [i for i in data_obj['schema'].keys() if data_obj['schema'][i] == 'numeric' and i in variables]
    factor_vars = [i for i in data_obj['schema'].keys() if data_obj['schema'][i] == 'factor' and i in variables]

    data_obj['info']['variables'] = variables
    print variables
    data_obj['info']['numeric_vars'] = numeric_vars
    data_obj['info']['factor_vars'] = factor_vars
    data_obj['info']['id_name'] = id_name
    data_obj['info']['target_name'] = target_name
    if 'fill_dict' in learn_conf_dict['preprocess_conf'].keys():
        fill_dict = learn_conf_dict['preprocess_conf']['fill_dict']
        data_obj['info']['fill_dict'] = fill_dict if fill_dict is not None else dict()
    else:
        data_obj['info']['fill_dict'] = dict()

    # process id and target cols convert to factors
    print '\n--- process id label type convert to factor ---'
    data_obj['df'] = process_id_label_type(data_obj['df'], id_name, target_name)

    if target_name:
        label_dict = label_static(data_obj['df'][target_name])
        for label in label_dict.keys():
            report_str += 'original label ' + str(label) + ' amount: ' + str(label_dict[label]) + '\n'
        report_str += '\n'

    # cal statics
    # # set sampling for cal statics
    data_obj, numeric_statics_list, factor_statics_list = learn_cal_summary_statics(data_obj, learn_conf_dict)

    # cal missing values row wise
    data_obj = learn_cal_samples_miss(data_obj)

    # cal distinct levels amount for factors
    data_obj = learn_cal_vars_levels(data_obj, learn_conf_dict)

    # cal missing values col wise
    data_obj = learn_cal_vars_miss(data_obj, learn_conf_dict)

    # cal std values
    data_obj = learn_cal_vars_std(data_obj, learn_conf_dict)

    # delete rows(samples)
    data_obj = learn_del_samples(data_obj, learn_conf_dict)

    # delete cols(variables)
    data_obj = learn_del_vars(data_obj, learn_conf_dict, ori_nrows)
    report_str += 'removed variable: ' + str(data_obj['info']['remove_vars']) + '\n\n'

    nrows_1 = data_obj['df'].shape[0]
    ncols_1 = data_obj['df'].shape[1]
    report_str += 'after deleting\n'
    report_str += 'sample amount: ' + str(nrows_1) + '\n'
    report_str += 'variable amount: ' + str(ncols_1) + '\n\n'

    # under-sampling
    if target_name and learn_conf_dict['preprocess_conf']['sampling_method'] == 'undersampling':
        data_obj = learn_under_sampling(data_obj, learn_conf_dict)

    if target_name:
        report_str += 'after sampling\n'
        label_dict = label_static(data_obj['df'][target_name])
        for label in label_dict.keys():
            report_str += 'label ' + str(label) + ' amount: ' + str(label_dict[label]) + '\n'
        report_str += '\n'

    # fill missing values
    data_obj = learn_fill_missing_value(data_obj, learn_conf_dict)

    # cal iv value
    iv_list = list()
    woe_list = list()
    if target_name:
        data_obj, iv_list, woe_list = learn_cal_iv(data_obj, learn_conf_dict)

    # reduce levels amount
    # data_obj = learn_reduce_levels(data_obj, learn_conf_dict)

    return data_obj, report_str, numeric_statics_list, factor_statics_list, iv_list, woe_list


def learn_preprocess_export(sc, preprocess_info_dict, process_report, numeric_statics_list, factor_statics_list, iv_list, woe_list, learn_fold_path):
    """
    
    :return: 
    """
    dump_dict2json_hdfs(sc, os.path.join(learn_fold_path, 'learn_preprocess.json'), preprocess_info_dict)

    write_to_hdfs(sc,  os.path.join(learn_fold_path, 'preprocess_report.txt'), process_report)

    if numeric_statics_list is not None and len(numeric_statics_list) >= 2:
        dump_list2csv_hdfs(sc, numeric_statics_list, os.path.join(learn_fold_path, 'numeric_variables_summary.csv'))

    if factor_statics_list is not None and len(factor_statics_list) >= 2:
        dump_list2csv_hdfs(sc, factor_statics_list, os.path.join(learn_fold_path, 'factor_variables_summary.csv'))

    if iv_list is not None and len(iv_list) >= 2:
        dump_list2csv_hdfs(sc, iv_list, os.path.join(learn_fold_path, 'iv_values.csv'))
        dump_list2csv_hdfs(sc, woe_list, os.path.join(learn_fold_path, 'woe_values.csv'))


def learn_train(train_df, x, y, learn_conf_dict):
    """
    train a model on entire data with the best parameters
    :return: 
    """

    algo = learn_conf_dict['train_conf']['algorithm']

    if algo in ['DL', 'LR', 'GBM', 'NB', 'RF']:

        trained_model, xval, valid = learn_train_single_classification_model(train_df=train_df, x=x, y=y,
                                                                             learn_conf_dict=learn_conf_dict)

        model_properties_dict, model_properties_str, cv_summary_lst, varimp_lst, coef_lst \
            = export_classification_model_properties_report(algo, trained_model=trained_model, xval=xval, valid=valid)

        metrics_obj = get_model_metric_obj(trained_model, valid=valid, xval=xval)
        model_metrics_dict, model_metrics_str, gains_lift_lst, thresholds_metrics_100_lst \
            = export_classification_model_metrics_report(metrics_obj=metrics_obj)

    elif algo == 'Stacking':
        trained_model = learn_train_stacking(train_df=train_df, x=x, y=y, learn_conf_dict=learn_conf_dict)

        model_properties_dict, model_properties_str, cv_summary_lst, varimp_lst, coef_lst \
            = export_classification_model_properties_report(algo, trained_model=trained_model, xval=False, valid=True)

        metrics_obj = get_model_metric_obj(trained_model, xval=False, valid=True)
        model_metrics_dict, model_metrics_str, gains_lift_lst, thresholds_metrics_100_lst \
            = export_classification_model_metrics_report(metrics_obj=metrics_obj)

    elif algo == 'KM':
        trained_model, xval, valid = learn_train_kmeans(train_df, x, learn_conf_dict)

        model_properties_dict, model_properties_str, model_metrics_dict \
            = export_kmeans_model_propertyies_report(trained_model=trained_model, xval=xval, valid=valid)
        model_metrics_str = ''
        cv_summary_lst, varimp_lst, coef_lst, gains_lift_lst, thresholds_metrics_100_lst \
            = None, None, None, None, None

    else:
        return None

    return trained_model, model_properties_dict, model_properties_str, model_metrics_dict, model_metrics_str, \
           cv_summary_lst, varimp_lst, coef_lst, gains_lift_lst, thresholds_metrics_100_lst


def learn_model_save(sc, model, learn_conf_dict):
    """
    
    :param sc: 
    :param model: 
    :param learn_fold_path: 
    :return: 
    """
    learn_fold_path = learn_conf_dict['learn_fold_path']
    learn_conf_dict['model_path'] = save_model(model, learn_fold_path)

    mojo_file_path = os.path.join(learn_fold_path, 'mojo')
    # pojo_file_path = os.path.join(learn_fold_path, 'pojo')
    make_dir(sc, mojo_file_path)
    # make_dir(sc, pojo_file_path)
    learn_conf_dict['mojo_path'] = save_model(model, mojo_file_path, model_type='mojo')
    # learn_conf_dict['pojo_path'] = save_model(model, learn_conf_dict['model_path'], model_type='pojo')

    return learn_conf_dict


def learn_train_report_export(sc, learn_conf_dict, model_properties_dict, model_properties_str,
                              model_metrics_dict, model_metrics_str,
                              cv_summary_lst, varimp_lst, coef_lst,
                              gains_lift_lst, thresholds_metrics_100_lst,
                              learn_fold_path):
    """
    
    :return: 
    """
    dump_dict2json_hdfs(sc, os.path.join(learn_fold_path, 'learn_configuration.json'), learn_conf_dict)
    dump_dict2json_hdfs(sc, os.path.join(learn_fold_path, 'model_properties.json'), model_properties_dict)
    dump_dict2json_hdfs(sc, os.path.join(learn_fold_path, 'model_metrics.json'), model_metrics_dict)

    write_to_hdfs(sc, os.path.join(learn_fold_path, 'model_properties.txt'), model_properties_str)
    write_to_hdfs(sc, os.path.join(learn_fold_path, 'model_metrics.txt'), model_metrics_str)

    for i in [[cv_summary_lst, 'cross-validation-summary.csv'],
              [varimp_lst, 'variable-importance.csv'],
              [coef_lst, 'variable-coefficients.csv'],
              [gains_lift_lst, 'gain-lift-table.csv'],
              [thresholds_metrics_100_lst, 'metrics-on-100-thresholds.csv']]:
        if isinstance(i[0], list) and len(i[0]) > 0:
            dump_list2csv_hdfs(sc, i[0], os.path.join(learn_fold_path, i[1]))


def learn_main(spark, config_filename):
    """
    build learning process
    
    preprocess: 
     1. choose if calculating statics ?
     2. if so, sampling or entire ?
     3. if sampling after undersampling ?
    
    
    :return: 
    """
    print "SuperAtom 20180125"
    sc = spark.sparkContext

    # parse configuration file
    learn_conf_dict, type_dict, col_name_lst = learn_configure(config_filename)
    learn_fold_path = learn_conf_dict['learn_fold_path']

    # create learn directory
    make_dir(sc, learn_fold_path)

    # import data set as a data frame and conduct  data audit
    data_obj = learn_data_import_audit(spark, learn_conf_dict, type_dict)

    # conduct preprocess
    data_obj, process_report, numeric_statics_list, factor_statics_list, iv_list, woe_list \
        = learn_preprocess(data_obj, learn_conf_dict)

    # preprocess report to hdfs
    preprocess_info_dict = data_obj['info']
    learn_preprocess_export(sc, preprocess_info_dict, process_report, numeric_statics_list, factor_statics_list,
                            iv_list, woe_list, learn_fold_path)

    # conduct model training
    trained_model, model_properties_dict, model_properties_str, model_metrics_dict, model_metrics_str, \
    cv_summary_lst, varimp_lst, coef_lst, gains_lift_lst, thresholds_metrics_100_lst \
        = learn_train(data_obj['df'],
                      x=data_obj['info']['variables'],
                      y=data_obj['info']['target_name'],
                      learn_conf_dict=learn_conf_dict)

    # save model
    learn_conf_dict = learn_model_save(sc, trained_model, learn_conf_dict)

    print '------ model_metrics_dict ------'
    print model_metrics_dict

    # save model and report to hdfs
    learn_train_report_export(sc, learn_conf_dict, model_properties_dict, model_properties_str,
                              model_metrics_dict, model_metrics_str, cv_summary_lst,
                              varimp_lst, coef_lst, gains_lift_lst, thresholds_metrics_100_lst,
                              learn_fold_path)

