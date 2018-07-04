# -*- coding:utf-8 -*-

# The functions of this module include:
#    1. convert a h2o two dimension table into a data frame and a list of lists;
#    2. get model summary info;
#    3. get actual values of params used in building final model;
#    4. get summary of a cross validation model;
#    5. get importance of variables from the model;
#    6. get coefficients of variables of a linear model;
#    7. get a h2o metric object from a trained model;
#    8. get scores of synthetical metrics;
#    9. get metrics' scores based on specific thresholds;
#    10. get metric' best score and its corresponding threshold;
#    11. get the gains lift table;
#    12. get a table of max scores of criteria metrics and their corresponding thresholds;
#    13. get confusion matrix based on a specified threshold;
#    14. calculate ks value;
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
    df_list = [col_names]
    df_list.extend(value_lst)

    return two_dim_df, df_list


# ============================
# model info
# ============================


def model_summary(trained_model):
    """
    get model summary info
    :param trained_model: a trained model
    :return: a summary in frame format, a summary in list format 
    """
    summary_t = trained_model.summary()
    summary_df, summary_lst = twodim_table_2_frame(summary_t)

    return summary_df, summary_lst


def get_actual_params(trained_model, param_lst=None):
    """
    get actual values of params used in building final model
    :param trained_model: a trained model
    :param param_lst: a list of parameters
    :return: a dict with params and their actual values used in building final model
    """
    param_actual_dict = dict()
    if param_lst is not None and len(param_lst) > 0:
        for param in param_lst:
            param_actual_dict[param] = trained_model.params[param]['actual']
    else:
        for param in trained_model.params.keys():
            param_actual_dict[param] = trained_model.params[param]['actual']

    return param_actual_dict


def cross_validation_summary(cv_model):
    """
    get summary of a cross validation model
    :param cv_model: a cross validation model
    :return: a summary in frame format, a summary in list format 
    """
    cv_summary_t = cv_model.cross_validation_metrics_summary()
    cv_summary_df, cv_summary_lst = twodim_table_2_frame(cv_summary_t)

    return cv_summary_df, cv_summary_lst


def var_important(trained_model):
    """
    get importance of variables from the model
    :param trained_model: a trained model
    :return: vars' importance in frame format, vars' importance in list format 
    """
    varimp_t = trained_model._model_json["output"]["variable_importances"]
    varimp_df, varimp_lst = twodim_table_2_frame(varimp_t)

    return varimp_df, varimp_lst


def linear_coef(trained_model):
    """
    get coefficients of variables of a linear model
    :param trained_model: a trained model
    :return: coefficients in frame format, coefficients in list format 
    """
    coef_t = trained_model._model_json["output"]["coefficients_table"]
    coef_df, coef_lst = twodim_table_2_frame(coef_t)

    return coef_df, coef_lst

# ============================
# metric info
# ============================


def get_model_metric_obj(trained_model, train=None, valid=None, xval=None):
    """
    get a h2o metric object from a trained model
    :param trained_model: a trained model
    :param train: whether metrics for training data
    :param valid: whether metrics for validation data
    :param xval: whether metrics for cross-validation data
    :return: a h2o metric object
    """
    if train:
        return trained_model._model_json["output"]["training_metrics"]
    elif valid:
        return trained_model._model_json["output"]["validation_metrics"]
    elif xval:
        return trained_model._model_json["output"]["cross_validation_metrics"]
    else:
        return None


def synthetical_metrics(metrics_obj):
    """
    get scores of synthetical metrics  like MSE, RMSE, logloss,  AIC, AUC, Gini, r2 and so on
    :param metrics_obj: a h2o metric object
    :return: a dict with metrics and their scores
    """
    metric_dict = {}
    for i in metrics_obj._metric_json.keys():
        if isinstance(metrics_obj._metric_json[i], float) \
                or isinstance(metrics_obj._metric_json[i], int):
            metric_dict[i] = metrics_obj._metric_json[i]

    return metric_dict


def find_metrics_by_threshold(metrics_obj, thresholds=None):
    """
    threshold, f1, f2, f0point5, accuracy, precision, recall, specificity, absolute_mcc, min_per_class_accuracy,
    mean_per_class_accuracy, tns, fns, fps, tps, tnr, fnr, fpr, tpr, idx
    :param metrics_obj:
    :param metrics:
    :param thresholds:
    :return:
    """
    thresholds_scores_t = metrics_obj._metric_json['thresholds_and_metric_scores']
    thresholds_scores_df, _ = twodim_table_2_frame(thresholds_scores_t)
    thresholds_scores_df.sort(by='threshold', ascending=[False])
    metrics_dict = dict()
    for threshold in thresholds:
        threshold = round(threshold, 2)
        idx = metrics_obj.find_idx_by_threshold(threshold)
        threshold_by_idx = float(h2o.as_list(thresholds_scores_df[thresholds_scores_df['idx'] == idx, ['threshold']]
                                             , use_pandas=False, header=False)[0][0])
        if threshold_by_idx >= threshold:
            metrics_dict[threshold] = dict(zip(*(h2o.as_list(thresholds_scores_df[thresholds_scores_df['idx'] == idx, :],
                                                             use_pandas=False))))
            for key in metrics_dict[threshold].keys():
                if metrics_dict[threshold][key] is not None and len(metrics_dict[threshold][key]) > 0:
                    metrics_dict[threshold][key] = float(metrics_dict[threshold][key])
                else:
                    metrics_dict[threshold][key] = 0
        else:
            if idx == 0:
                idx_1_metrics = dict(zip(*(h2o.as_list(thresholds_scores_df[thresholds_scores_df['idx'] == 0, :],
                                                       use_pandas=False))))
                fps, tps, fpr, tpr, precision, recall, f1, f2, f0point5 = 0, 0, 0, 0, 0, 0, 0, 0, 0
                tnr, fnr, specificity = 1, 1, 1
                sample_sum = int(idx_1_metrics['tns']) + int(idx_1_metrics['fns']) \
                             + int(idx_1_metrics['tps']) + int(idx_1_metrics['fps'])
                tns = int(idx_1_metrics['tns']) + int(idx_1_metrics['fps'])
                fns = sample_sum - tns
                accuracy = (tns + tps) / sample_sum
                absolute_mcc = None
                min_per_class_accuracy = None
                mean_per_class_accuracy = None
                metrics_dict[threshold] = dict({'threshold': 1, 'f1': f1, 'f2': f2, 'f0point5': f0point5,
                                                'accuracy': accuracy, 'precision': precision, 'recall': recall,
                                                'specificity': specificity, 'absolute_mcc': absolute_mcc,
                                                'min_per_class_accuracy': min_per_class_accuracy,
                                                'mean_per_class_accuracy': mean_per_class_accuracy, 'tns': tns,
                                                'fns': fns, 'fps': fps, 'tps': tps, 'tnr': tnr, 'fnr': fnr, 'fpr': fpr,
                                                'tpr': tpr, 'idx': -1})
            else:
                metrics_dict[threshold] = dict(zip(*(h2o.as_list(thresholds_scores_df[thresholds_scores_df['idx'] == int(idx - 1)],
                                                                 use_pandas=False))))
                for key in metrics_dict[threshold].keys():
                    if metrics_dict[threshold][key] is not None and len(metrics_dict[threshold][key]) > 0:
                        metrics_dict[threshold][key] = float(metrics_dict[threshold][key])
                    else:
                        metrics_dict[threshold][key] = 0

    return metrics_dict


# def metrics_by_threshold(metrics_obj, metric_threshold_dict=None):
#     """
#     get metrics' scores based on specific thresholds
#     ['f1', 'f2', 'f0point5', 'accuracy', 'precision', 'recall', 'tpr', 'tnr', 'fnr', 'fpr',
#     'sensitivity', 'fallout', 'missrate', 'specificity', 'mcc']
#     :param metrics_obj: a h2o metric object
#     :param metric_threshold_dict: a dict with metrics and their specific thresholds
#     :return: a dict  with metrics and their scores
#     """
#     metrics_lst = ['f1', 'f2', 'f0point5', 'accuracy', 'precision', 'recall', 'tpr', 'tnr', 'fnr', 'fpr',
#                    'sensitivity', 'fallout', 'missrate', 'specificity', 'mcc']
#     if metric_threshold_dict is None:
#         metric_threshold_dict = dict(zip(metrics_lst, [0.5] * len(metrics_lst)))
#     metric_dict = dict()
#     special_map = {'recall': 'tpr', 'sensitivity': 'tpr', 'fallout': 'fpr',
#                    'missrate': 'fnr', 'specificity': 'tnr', 'mcc': 'absolute_mcc'}
#     for metric in metric_threshold_dict.keys():
#         threshold = metric_threshold_dict[metric]
#         if metric in special_map.keys():
#             mapped_metric = special_map[metric]
#         else:
#             mapped_metric = metric
#         try:
#             metric_val = metrics_obj.metric(metric=mapped_metric, thresholds=[threshold])[0][1]
#         except ValueError:
#             metric_val = None
#         metric_dict[metric] = metric_val
#
#     return metric_dict


def max_metric_threshold(metrics_obj, metric_lst=None):
    """
    get metric' best score and its corresponding threshold
    :param metrics_obj: a h2o metric object
    :param metric_lst: a list of metrics
    :return: a dict with metrics and their thresholds
    """
    default_metric_lst = ["min_per_class_accuracy", "absolute_mcc", "precision", "recall",
                          "specificity", "accuracy", "f0point5", "f2", "f1", "mean_per_class_accuracy"]
    metric_dict = dict()
    if metric_lst is not None and len(metric_lst) > 0:
        for metric in metric_lst:
            threshold = metrics_obj.find_threshold_by_max_metric(metric)
            metric_dict[metric] = threshold
    else:
        for metric in default_metric_lst:
            threshold = metrics_obj.find_threshold_by_max_metric(metric)
            metric_dict[metric] = threshold

    return metric_dict


def gains_lift(metrics_obj):
    """
    get the gains lift table
    :param metrics_obj: a h2o metric object
    :return: a gains-lift table in frame format, a gains-lift table in list format
    """
    gains_lift_t = metrics_obj.gains_lift()
    gains_lift_df, gains_lift_lst = twodim_table_2_frame(gains_lift_t)

    return gains_lift_df, gains_lift_lst


def max_criteria_metric(metrics_obj):
    """
    get a table of max scores of criteria metrics and their corresponding thresholds
    :param metrics_obj: a h2o metric object
    :return: a max criteria metric scores and thresholds in frame format, and in list format
    """
    max_criteria_metric_t = metrics_obj._metric_json["max_criteria_and_metric_scores"]
    max_criteria_metric_df, max_criteria_metric_lst = twodim_table_2_frame(max_criteria_metric_t)

    return max_criteria_metric_df, max_criteria_metric_lst


def confusion_matrix(metrics_obj, threshold_lst=None, metrics_dict=None):
    """
    get confusion matrix based on a specified threshold
    :param metrics_obj: a h2o metric object
    :param threshold_lst: a threshold value
    :return: a confusion matrix in frame format, a confusion matrix in list format
    """
    if isinstance(threshold_lst, list):
        confusion_matrix_dict = dict()
        if metrics_dict is None:
            metrics_dict = find_metrics_by_threshold(metrics_obj, threshold_lst)
        if threshold_lst is None:
            threshold_lst = list(metrics_dict.keys())

        c_matrix = metrics_obj.confusion_matrix(thresholds=[0.5]).table
        _, c_matrix_lst = twodim_table_2_frame(c_matrix)
        neg_label, pos_label = c_matrix_lst[0][1], c_matrix_lst[0][2]

        for threshold in threshold_lst:
            threshold = round(threshold, 2)
            tns = metrics_dict[threshold]['tns']
            fps = metrics_dict[threshold]['fps']
            fns = metrics_dict[threshold]['fns']
            tps = metrics_dict[threshold]['tps']

            confusion_matrix_dict[threshold] = [['', neg_label, pos_label, 'Error', 'Rate'],
                                                [neg_label, tns, fps, fns*1.0 / (tns + fps),
                                                 str('(') + str(fns) + str(' / ') + str(tns + fps) + str(')')],
                                                [pos_label, fns, tps, fps * 1.0 / (tps + fns),
                                                 str('(') + str(fps) + str(' / ') + str(tps + fns) + str(')')],
                                                ['Total', tns + fns, fps + tps, (fps + fns) * 1.0 / (tns + fns + fps + tps),
                                                 str('(') + str(fps + fns) + str(' / ') + str(tns + fns + fps + tps) + str(')')]]

        return confusion_matrix_dict
    else:
        return None


def cal_ks_val(metrics_obj):
    """
    calculate ks value : KS=max(TPR-FPR)
    :param metrics_obj: a h2o metric object
    :return: a simplified threshold metric scores table, ks value , and its corresponding threshold
    """
    thresholds_scores_t = metrics_obj._metric_json['thresholds_and_metric_scores']
    thresholds_scores_df, _ = twodim_table_2_frame(thresholds_scores_t)
    thresholds_scores_df['tpr_fpr'] = thresholds_scores_df['tpr'] - thresholds_scores_df['fpr']
    ks_val = thresholds_scores_df['tpr_fpr'].max()
    ks_idx = thresholds_scores_df['tpr_fpr'].idxmax()[0, 0]
    ks_threshold = thresholds_scores_df[thresholds_scores_df['idx'] == ks_idx]['threshold'][0, 0]
    sample_num = thresholds_scores_df.shape[0]
    output_df_nrow = 20
    output_df_nrow_half = output_df_nrow / 2
    if sample_num <= (output_df_nrow + 10):
        thresholds_scores_df_s = thresholds_scores_df
    else:
        if ks_idx <= output_df_nrow_half:
            idx_lst_1 = list(range(ks_idx))
            inter = 1.0 * (sample_num - ks_idx) / (output_df_nrow - ks_idx)
            idx_lst_2 = [int(ks_idx + (i + 1) * inter) for i in range(output_df_nrow - ks_idx)]
            idx_lst = idx_lst_1 + [ks_idx] + idx_lst_2
        elif (sample_num - ks_idx) <= output_df_nrow_half:
            idx_lst_2 = list(range(ks_idx + 1, sample_num))
            inter = 1.0 * ks_idx / (output_df_nrow - (sample_num - ks_idx))
            idx_lst_1 = [int(i * inter) for i in range(output_df_nrow - (sample_num - ks_idx))]
            idx_lst = idx_lst_1 + [ks_idx] + idx_lst_2
        else:
            inter_1 = 1.0 * ks_idx / output_df_nrow_half
            idx_lst_1 = [int(i * inter_1) for i in range(output_df_nrow_half)]
            inter_2 = 1.0 * (sample_num - ks_idx) / output_df_nrow_half
            idx_lst_2 = [int(ks_idx + (i + 1) * inter_2) for i in range(output_df_nrow_half)]
            idx_lst = idx_lst_1 + [ks_idx] + idx_lst_2

        thresholds_scores_df_s = thresholds_scores_df[idx_lst, :].asnumeric()

    thresholds_scores_df_s_lst = h2o.as_list(thresholds_scores_df_s)

    return thresholds_scores_df_s, thresholds_scores_df_s_lst, ks_val, ks_threshold


def thresholds_metrics_100(metrics_obj):
    """

    :param metrics_obj:
    :return:
    """
    # metrics_lst = ['f1', 'f2', 'f0point5', 'accuracy', 'precision', 'recall', 'tpr', 'tnr', 'fnr', 'fpr',
    #                'tps', 'tns', 'fns', 'fps', 'sensitivity', 'specificity']
    # thresholds_metrics_100_lst = list()
    # thresholds_metrics_100_lst.append(['threshold'] + metrics_lst)
    # for i in range(100):
    #     threshold = i * 0.01
    #     metric_threshold_dict = dict(zip(metrics_lst, [threshold] * len(metrics_lst)))
    #     metric_dict = metrics_by_threshold(metrics_obj, metric_threshold_dict=metric_threshold_dict)
    #     thresholds_metrics_100_lst.append([threshold] + [metric_dict[j] for j in metrics_lst])

    thresholds_metrics_100_dict = find_metrics_by_threshold(metrics_obj, [0.01 * i for i in range(101)])
    thresholds_metrics_100_lst = list()
    headers = list((thresholds_metrics_100_dict.values())[0].keys())
    thresholds_metrics_100_lst.append(headers)
    for threshold in thresholds_metrics_100_dict.keys():
        threshold = round(threshold, 2)
        thresholds_metrics_100_dict[threshold]['threshold'] = threshold
        thresholds_metrics_100_lst.append([thresholds_metrics_100_dict[threshold][metric] for metric in headers])

    ks = 0
    ks_threshold = 0
    for threshold in thresholds_metrics_100_dict.keys():
        threshold = round(threshold, 2)
        tpr_fpr = thresholds_metrics_100_dict[threshold]['tpr'] - thresholds_metrics_100_dict[threshold]['fpr']
        if tpr_fpr > ks:
            ks = tpr_fpr
            ks_threshold = threshold

    return thresholds_metrics_100_dict, thresholds_metrics_100_lst, ks, ks_threshold


def score_group_metrics_10(metrics_obj, thresholds_metrics_100_dict=None):
    """

    :param metrics_obj:
    :return:
    """
    metrics = ['tps', 'fps', 'tns', 'fns', 'recall', 'precision', 'accuracy', 'specificity']
    if thresholds_metrics_100_dict is None:
        thresholds_metrics_100_dict, thresholds_metrics_100_lst, ks, ks_threshold = thresholds_metrics_100(metrics_obj)
    # print thresholds_metrics_100_lst

    thresholds_metrics_10_dict = dict()
    for threshold in thresholds_metrics_100_dict.keys():
        if int(threshold * 100) % 10 == 0:
            #fixed_threshold = int(threshold * 10) * 0.1
            thresholds_metrics_10_dict[round(threshold, 1)] = thresholds_metrics_100_dict[threshold]

    # threshold_ind = thresholds_metrics_100_lst[0].index('threshold')
    # thresholds_metrics_10_lst = [thresholds_metrics_100_lst[0]] \
    #                              + [i for i in thresholds_metrics_100_lst[1:] if int(i[threshold_ind] * 100) % 10 == 0]
    #
    # thresholds_metrics_10_dict = dict()
    # print thresholds_metrics_10_lst
    # for i in thresholds_metrics_10_lst[1:]:
    #     thresholds_metrics_10_dict[i[0]] = dict(zip(thresholds_metrics_10_lst[0][:], i[:]))

    print thresholds_metrics_10_dict
    for i in range(10):
        upper_threshold = round(1 - i*0.1, 1)
        up_dict = thresholds_metrics_10_dict[upper_threshold]
        lower_threshold = round(1 - (i + 1) * 0.1, 1)
        low_dict = thresholds_metrics_10_dict[lower_threshold]
        tmp_dict = dict()
        tmp_dict['sum'] = low_dict['tps'] + low_dict['fps'] + low_dict['tns'] + low_dict['fns']

        tmp_dict['tps'] = low_dict['tps'] - up_dict['tps']
        tmp_dict['fps'] = low_dict['fps'] - up_dict['fps']

        tmp_dict['tns'] = up_dict['tns'] - low_dict['tns']
        tmp_dict['fns'] = up_dict['fns'] - low_dict['fns']

        thresholds_metrics_10_dict[str(lower_threshold) + '-' + str(upper_threshold)] = tmp_dict

    score_group_threshold_10_dict = dict()
    for i in [m*0.1 for m in range(11)]:
        grouped_metrics_table = list()
        header = ['score_bins'] + metrics
        grouped_metrics_table.append(header)
        for j in [k*0.1 for k in range(10)]:
            grouped_metrics_dict = dict()
            lower_threshold = round(j, 1)
            upper_threshold = round(j + 0.1, 1)
            tmp_1 = thresholds_metrics_10_dict[str(lower_threshold) + '-' + str(upper_threshold)]
            if j >= i:
                grouped_metrics_dict['fraction'] = (tmp_1['tps'] + tmp_1['fps']) / tmp_1['sum']
                grouped_metrics_dict['tps'] = tmp_1['tps']
                grouped_metrics_dict['fps'] = tmp_1['fps']
                grouped_metrics_dict['tns'] = 0
                grouped_metrics_dict['fns'] = 0
                grouped_metrics_dict['recall'] = 1 if tmp_1['tps'] != 0 else 0
                grouped_metrics_dict['precision'] = tmp_1['tps'] / (tmp_1['tps'] + tmp_1['fps']) if tmp_1['tps'] != 0 \
                    else 0
                grouped_metrics_dict['accuracy'] = grouped_metrics_dict['precision']
                grouped_metrics_dict['specificity'] = 0
            else:
                grouped_metrics_dict['fraction'] = (tmp_1['tns'] + tmp_1['fns']) / tmp_1['sum']
                grouped_metrics_dict['tps'] = 0
                grouped_metrics_dict['fps'] = 0
                grouped_metrics_dict['tns'] = tmp_1['tns']
                grouped_metrics_dict['fns'] = tmp_1['fns']
                grouped_metrics_dict['recall'] = 0
                grouped_metrics_dict['precision'] = 0
                grouped_metrics_dict['accuracy'] = tmp_1['tns'] / (tmp_1['tns'] + tmp_1['fns']) if tmp_1['tns'] != 0 \
                    else 0
                grouped_metrics_dict['specificity'] = 1 if tmp_1['tns'] != 0 else 0

            grouped_metrics_lst = [str(lower_threshold) + '-' + str(upper_threshold)] \
                                  + [grouped_metrics_dict[z] for z in metrics]

            grouped_metrics_table.append(grouped_metrics_lst)
            # thresholds_dict[str(lower_threshold) + '-' + str(upper_threshold)] = grouped_metrics_lst

        score_group_threshold_10_dict[round(i, 1)] = grouped_metrics_table

    return score_group_threshold_10_dict


def topn_metrics_10(metrics_obj):
    """

    :param metrics_obj:
    :return:
    """
    metrics = ['tps', 'fps', 'tns', 'fns', 'recall', 'precision', 'accuracy', 'specificity']
    thresholds_scores_t = metrics_obj._metric_json['thresholds_and_metric_scores']
    thresholds_scores_df, _ = twodim_table_2_frame(thresholds_scores_t)
    thresholds_scores_df['cum_ps'] = thresholds_scores_df['tps'] + thresholds_scores_df['fps']
    cum_ps_lst = h2o.as_list(thresholds_scores_df['cum_ps'], use_pandas=False, header=False)
    cum_ps_lst = [int(i[0]) for i in cum_ps_lst]
    cum_ps_lst = sorted(cum_ps_lst)
    amount = max(cum_ps_lst)
    i = 1
    threshold_cum_ps_lst = list()
    start_ind = 0
    while i <= 10:
        mark = 0
        for j in cum_ps_lst[start_ind:]:
            if j >= int(i * amount / 10) and len(threshold_cum_ps_lst) < i:
                threshold_cum_ps_lst.append(j)
                i += 1
                mark = j
                break
        start_ind = mark

    topn_metrics_10_lst = list()
    header = ['score_topN'] + metrics
    topn_metrics_10_lst.append(header)
    i = 1
    for cum_ps in threshold_cum_ps_lst:
        tmp_metric_df = thresholds_scores_df[thresholds_scores_df['cum_ps'] == cum_ps]
        tmp_metric_dict = dict(zip(*(h2o.as_list(tmp_metric_df, use_pandas=False))))
        metric_lst = ['Top' + str(10*i) + '%']
        i += 1
        for metric in metrics:
            metric_lst.append(tmp_metric_dict[metric])

        topn_metrics_10_lst.append(metric_lst)

    return topn_metrics_10_lst
