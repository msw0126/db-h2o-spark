# -*- coding:utf-8 -*-

# The functions of this module include(model properties for clustering algorithm, like k-means):
#    1. convert a h2o two dimension table into a data frame and a list of lists;
#    2. get model properties based on specific set (train set, validation set or cross validation set);
#    3. get model properties (not base on specific data set);
#    4. get metric scores on test set;
#

import h2o


def twodim_table_2_frame(two_dim_t):
    """
    convert a h2o two dimension table into 2 formats:
        1. h2o data frame
        2. a list of lists
    :param two_dim_t: a h2o two dimension table
    :return: a frame format table, a list format table
    """
    value_lst = two_dim_t._cell_values
    col_names = two_dim_t._col_header
    two_dim_df = h2o.H2OFrame(value_lst, column_names=col_names)

    return two_dim_df, [col_names, value_lst]


def get_km_model_properties_by_set(trained_model, train=None, xval=None, valid=None):
    """
    get model's property parameters based on specific set(training data, validation data, or cross-validation data):
    1. betweenss: the between cluster sum-of-square error
    2. tot_withinss: the total within cluster sum of square error 
    3. size: the sizes of each cluster
    4. totss: the total sum-of-square error to grand mean
    5. withinss: the within cluster sum of squares for each cluster
    6. centroid_stats: centroid statistics for each cluster
    :param trained_model: a trained model
    :param properties_dict: 
    :return: a dict with property parameters and their corresponding value based on a specific set
    """
    result_dict = dict()
    if train is None and xval is None and valid is None:
        properties_dict = dict(zip(['betweenss', 'tot_withinss', 'size', 'totss', 'withinss', 'centroid_stats'],
                                   [['train', 'valid', 'xval'] for i in range(6)]))
    else:
        set_lst = list()
        if train:
            set_lst.append('train')
        if xval:
            set_lst.append('xval')
        if valid:
            set_lst.append('valid')
        properties_dict = dict(zip(['betweenss', 'tot_withinss', 'size', 'totss', 'withinss', 'centroid_stats'],
                                   [set_lst for i in range(6)]))

    for target in properties_dict.keys():
        if target in ['betweenss', 'tot_withinss', 'size', 'totss', 'withinss']:
            func_expr = 'trained_model.' + str(target) + '(' \
                        + ','.join([str(i) + '=True' for i in properties_dict[target]]) + ')'
            result_dict[target] = eval(func_expr)

        if target == 'centroid_stats':
            result_dict[target] = dict()
            for set_type in properties_dict[target]:
                if set_type == 'train':
                    train_metric_obj = trained_model._model_json["output"]["training_metrics"]
                    if train_metric_obj is not None:
                        _, result_dict[target][set_type] \
                            = twodim_table_2_frame(train_metric_obj._metric_json["centroid_stats"])

                elif set_type == 'valid':
                    valid_metric_obj = trained_model._model_json["output"]["validation_metrics"]
                    if valid_metric_obj is not None:
                        _, result_dict[target][set_type] \
                            = twodim_table_2_frame(valid_metric_obj._metric_json['centroid_stats'])

                elif set_type == 'xval':
                    xval_metric_obj = trained_model._model_json["output"]["cross_validation_metrics"]
                    if xval_metric_obj is not None:
                        _, result_dict[target][set_type] \
                            = twodim_table_2_frame(xval_metric_obj._metric_json['centroid_stats'])
                else:
                    result_dict[target][set_type] = None

    return result_dict


def get_km_model_properties(trained_model):
    """
    get model's property parameters:
    1. params: actual parameters used when modeling
    2. centers: centers of the cluster model
    3. num_iterations: iteration number for training model
    4. centers_std: centroid statistics for each cluster
    :param trained_model: a trained model
    :return: a dict with property parameters and their corresponding value
    """
    result_dict = dict()
    for target in ['params', 'centers', 'num_iterations', 'centers_std']:
        if target == 'params':
            for key in ['standardize', 'k', 'init', 'estimate_k', 'max_iterations']:
                result_dict[target] = trained_model.params[key]['actual']
        else:
            if target == 'centers_std':
                result_dict[target] = trained_model._model_json["output"]['centers_std']
            else:
                result_dict[target] = eval('trained_model.' + str(target) + '()')

    return result_dict

# ====================
# metric info
# ====================


def get_km_test_metrics(metrics_obj):
    """
    get test metrics' scores of clustering model
    betweenss: the between cluster sum-of-square error
    tot_withinss: the total within cluster sum of square error
    totss: the total sum-of-square error to grand mean
    centroid_stats: centroid statistics for each cluster
    :param metrics_obj: a metric object
    :return: a dict with metrics and their corresponding scores
    """
    metrics_dict = dict()
    metrics_dict['betweenss'] = metrics_obj.betweenss()
    metrics_dict['tot_withinss'] = metrics_obj.tot_withinss()
    metrics_dict['totss'] = metrics_obj.totss()
    metrics_dict['centroid_stats'] = twodim_table_2_frame(metrics_obj._metric_json['centroid_stats'])[1]

    return metrics_dict
