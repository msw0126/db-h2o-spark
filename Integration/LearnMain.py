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
        # data_df: 是表的所有内容
        data_df = hive_to_hdfs(spark, learn_conf_dict['hive_table'])
        # data_obj： {'df': data_df, 'schema': type_dict, 'info': dict()}
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
    # 判断训练数据的表，列是否大于2，行是否大于50行
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
    # data_obj['info']： {'ori_type_dict': {'v18': 'numeric', 'v19': 'factor', 'v12': 'factor', 'v13': 'numeric', 'v10': 'factor', 'v11': 'numeric', 'v16': 'numeric', 'v17': 'factor', 'v14': 'factor', 'v15': 'factor', 'id': 'numeric', 'v21': 'numeric', 'v20': 'factor', 'v1': 'factor', 'v2': 'numeric', 'v3': 'factor', 'v4': 'factor', 'v5': 'numeric', 'v6': 'factor', 'v7': 'factor', 'v8': 'numeric', 'v9': 'factor'}}
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
    """
    report_str = \
    id name: id
    target name: v21

    original sample amount: 1000
    original variable amount: 22
    """

    # 删除id和target字段
    variables = data_obj['df'].names
    if id_name:
        variables.remove(id_name)
    if target_name:
        variables.remove(target_name)

    # data_obj['schema'] {'v18': 'numeric', 'v19': 'factor', 'v12': 'factor', 'v13': 'numeric', 'v10': 'factor', 'v11': 'numeric', 'v16': 'numeric', 'v17': 'factor', 'v14': 'factor', 'v15': 'factor', 'id': 'numeric', 'v21': 'numeric', 'v20': 'factor', 'v1': 'factor', 'v2': 'numeric', 'v3': 'factor', 'v4': 'factor', 'v5': 'numeric', 'v6': 'factor', 'v7': 'factor', 'v8': 'numeric', 'v9': 'factor'}
    # 得到字段类型分别为numeric和factor的字段名列表
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
    # 把表的id、target字段转换为factor类型
    data_obj['df'] = process_id_label_type(data_obj['df'], id_name, target_name)

    if target_name:
        # lable_dict: {'1': 700, '0': 300}
        label_dict = label_static(data_obj['df'][target_name])
        for label in label_dict.keys():
            report_str += 'original label ' + str(label) + ' amount: ' + str(label_dict[label]) + '\n'
        report_str += '\n'


    # cal statics
    # # set sampling for cal statics
    """
    numeric_statics_list: [['variable', 'max', 'min', 'mean', 'sigma', 'median', 'missing_count'], [u'v18', 2.0, 1.0, 1.1549999999999991, 0.3620857717531942, 1.0, 0],
    factor_statics_list: [['variable', 'level_num', 'missing_count', 'most_freq_level', 'levels'], [u'v19', 2, 0, 'A191', u'A191|A192']
    """
    data_obj, numeric_statics_list, factor_statics_list = learn_cal_summary_statics(data_obj, learn_conf_dict)

    # cal missing values row wise
    # 每个样本的缺失值统计。data_obj['info']['row_miss_lst'] = row_miss_lst
    data_obj = learn_cal_samples_miss(data_obj)

    # cal distinct levels amount for factors(计算因子变量的级别数)
    # levels_dict[var] = data_obj['info']['factor_statics'][var]['levels']
    # data_obj['info']['levels_dict'] = levels_dict
    data_obj = learn_cal_vars_levels(data_obj, learn_conf_dict)

    # cal missing values col wise(增加每列与缺失值数量的字典)
    # data_obj['info']['col_miss_dict'] = col_miss_dict
    data_obj = learn_cal_vars_miss(data_obj, learn_conf_dict)

    # cal std values(计算数值变量的标准差)
    # data_obj['info']['col_std_dict'] = col_std_dict
    data_obj = learn_cal_vars_std(data_obj, learn_conf_dict)

    # delete rows(samples)(如果某一个样本的缺失值过多，就把这行样本删除)
    data_obj = learn_del_samples(data_obj, learn_conf_dict)

    # delete cols(variables)(由于几种原因，删除某些列)
    data_obj = learn_del_vars(data_obj, learn_conf_dict, ori_nrows)
    report_str += 'removed variable: ' + str(data_obj['info']['remove_vars']) + '\n\n'

    nrows_1 = data_obj['df'].shape[0]
    ncols_1 = data_obj['df'].shape[1]
    report_str += 'after deleting\n'
    report_str += 'sample amount: ' + str(nrows_1) + '\n'
    report_str += 'variable amount: ' + str(ncols_1) + '\n\n'

    # under-sampling(如果正负样本差别太大，会删除大部分的某样本比较多的数据，维持正负样本在正常比例。如果差异不大，会直接返回)
    if target_name and learn_conf_dict['preprocess_conf']['sampling_method'] == 'undersampling':
        data_obj = learn_under_sampling(data_obj, learn_conf_dict)

    if target_name:
        report_str += 'after sampling\n'
        label_dict = label_static(data_obj['df'][target_name])
        for label in label_dict.keys():
            report_str += 'label ' + str(label) + ' amount: ' + str(label_dict[label]) + '\n'
        report_str += '\n'

    # fill missing values（填充缺失值）
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
        # 训练模型
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

    # parse configuration file(解析配置文件)
    """
    learn_conf_dict: {'data_missing_symbol': ['null', 'Null', 'NULL', 'NaN', 'nan', 'Na', 'NA', 'N/A', 'None', 'NONE', '\\N', '', '?'], 'target_name': 'v21', 'train_conf': {'cv_k': 1, 'hparams': {'min_rows': [1, 2], 'ntrees': [50], 'sample_rate': [0.63], 'max_depth': [5, 10], 'col_sample_rate_per_tree': [1.0]}, 'algorithm': 'RF'}, 'hive_table': 'taoshu_db_input.german_credit', 'id_name': 'id', 'train_data_path': 'hdfs://node1:8020/taoshu/engine/work_dir/103/AtomLearn1/data.csv', 'data_sep_symbol': ',', 'data_types': None, 'preprocess_conf': {'cal_statics_sampling': False, 'max_sample_miss_prop': 0.95, 'fill_dict': None, 'sampling_method': 'undersampling', 'cal_iv': None, 'max_factor_levels_prop': 1000, 'unbalanced_cutoff': 5, 'cal_statics_universal': True, 'max_levels_amount': 200, 'max_variable_miss_prop': 0.9}, 'learn_fold_path': 'hdfs://node1:8020/taoshu/engine/work_dir/103/AtomLearn1/LEARN'}
    type_dict: {'v18': 'numeric', 'v19': 'factor', 'v12': 'factor', 'v13': 'numeric', 'v10': 'factor', 'v11': 'numeric', 'v16': 'numeric', 'v17': 'factor', 'v14': 'factor', 'v15': 'factor', 'id': 'numeric', 'v21': 'numeric', 'v20': 'factor', 'v1': 'factor', 'v2': 'numeric', 'v3': 'factor', 'v4': 'factor', 'v5': 'numeric', 'v6': 'factor', 'v7': 'factor', 'v8': 'numeric', 'v9': 'factor'}
    col_name_lst: ['id', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21']
    """
    learn_conf_dict, type_dict, col_name_lst = learn_configure(config_filename)
    learn_fold_path = learn_conf_dict['learn_fold_path']

    # create learn directory(创建learn的HDFS工作目录)
    make_dir(sc, learn_fold_path)

    # import data set as a data frame and conduct  data audit（导入数据，并对数据进行检查）
    # data_obj： {'df': data_df, 'schema': type_dict, 'info': dict()}， data_df： 是表格的所有数据，并按type进行排序
    data_obj = learn_data_import_audit(spark, learn_conf_dict, type_dict)

    # conduct preprocess(进行数据预处理)
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

