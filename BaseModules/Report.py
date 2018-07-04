
from ClassificationModelInfo import *
from ClusterModelInfo import *


def list_2_csv_string(content_list):
    """

    :return: 
    """
    content_str_csv = ''
    for i in range(len(content_list)):
        i_content = ','.join([str(col) for col in content_list[i]]) + '\n'
        content_str_csv += i_content
    return content_str_csv


def export_classification_model_properties_report(algo, trained_model, xval=None, valid=None):
    """

    :return: 
    """
    print '\n===== export model properties ====='
    model_properties_dict = dict()
    model_properties_str = '=============== model properties ===============\n\n'

    print '\n----- actual params -----'

    actual_params_dict = get_actual_params(trained_model)
    model_properties_dict['actual_params_dict'] = actual_params_dict

    model_properties_str += '***** actual parameters ***** \n'
    for param in actual_params_dict.keys():
        model_properties_str += (str(param) + ': ' + str(actual_params_dict[param]) + '\n')
    model_properties_str += '\n\n'

    print actual_params_dict

    print '\n----- model summary -----'
    if algo != 'Stacking':
        summary_df, summary_lst = model_summary(trained_model)
        model_properties_dict['summary_lst'] = summary_lst

        model_properties_str += '***** model summary ***** \n'
        model_properties_str += (list_2_csv_string(summary_lst) + '\n')
        model_properties_str += '\n\n'

        print summary_df, '\n', summary_lst

    cv_summary_lst = list()
    if xval:
        print '\n----- cross validation summary -----'
        cv_summary_df, cv_summary_lst = cross_validation_summary(trained_model)
        model_properties_dict['cv_summary_lst'] = cv_summary_lst

        model_properties_str += '***** cross-validation model summary ***** \n'
        model_properties_str += (list_2_csv_string(cv_summary_lst) + '\n')
        model_properties_str += '\n\n'

        print cv_summary_df, '\n', cv_summary_lst

    print '\n----- variable importance -----'
    varimp_lst = list()
    if algo in ['DL', 'GBM', 'RF']:
        varimp_df, varimp_lst = var_important(trained_model)
        model_properties_dict['varimp_lst'] = varimp_lst

        model_properties_str += '***** variable importance ***** \n'
        model_properties_str += (list_2_csv_string(varimp_lst) + '\n')
        model_properties_str += '\n\n'

        print varimp_df, '\n', varimp_lst

    print '\n----- linear coefficients -----'
    coef_lst = list()
    if algo == 'LR':
        coef_df, coef_lst = linear_coef(trained_model)
        model_properties_dict['coef_lst'] = coef_lst

        model_properties_str += '***** linear coefficients ***** \n'
        model_properties_str += (list_2_csv_string(coef_lst) + '\n')
        model_properties_str += '\n\n'

        print coef_df, '\n', coef_lst

    return model_properties_dict, model_properties_str, cv_summary_lst, varimp_lst, coef_lst


def export_classification_model_metrics_report(metrics_obj):
    """
    
    :return: 
    """
    print '\n===== export model metrics ====='
    model_metrics_dict = dict()
    model_metrics_str = '=============== model metrics ===============\n\n'

    print '\n----- synthetical metrics -----'
    synthetical_metric_dict = synthetical_metrics(metrics_obj)
    model_metrics_dict['synthetical_metric_dict'] = synthetical_metric_dict

    model_metrics_str += '***** synthetical metrics ***** \n'
    for metric in synthetical_metric_dict.keys():
        model_metrics_str += (str(metric) + ': '
                              + str(synthetical_metric_dict[metric]) + '\n')
    model_metrics_str += '\n\n'

    print synthetical_metric_dict

    print '\n----- metrics by threshold 0.5 -----'
    metric_threshold_dict = find_metrics_by_threshold(metrics_obj, [0.5])
    model_metrics_dict['metric_threshold_dict'] = metric_threshold_dict

    model_metrics_str += '***** metrics by threshold 0.5 ***** \n'
    for metric in metric_threshold_dict.keys():
        model_metrics_str += (str(metric) + ': '
                              + str(metric_threshold_dict[metric]) + '\n')
    model_metrics_str += '\n\n'

    print metric_threshold_dict

    print '\n----- thresholds by max metric scores -----'
    metric_max_score_dict = max_metric_threshold(metrics_obj)
    model_metrics_dict['metric_max_score_dict'] = metric_max_score_dict

    model_metrics_str += '***** thresholds by max metric scores ***** \n'
    for metric in metric_max_score_dict.keys():
        model_metrics_str += (str(metric) + ': '
                              + str(metric_max_score_dict[metric]) + '\n')
    model_metrics_str += '\n\n'

    print metric_max_score_dict

    print '\n----- gains / lift-----'
    gains_lift_df, gains_lift_lst = gains_lift(metrics_obj)
    model_metrics_dict['gains_lift_lst'] = gains_lift_lst

    model_metrics_str += '***** gains / lift ***** \n'
    model_metrics_str += (list_2_csv_string(gains_lift_lst) + '\n')
    model_metrics_str += '\n\n'

    print gains_lift_df

    print '\n----- max criteria metric scores thresholds table-----'
    max_criteria_metric_df, max_criteria_metric_lst = max_criteria_metric(metrics_obj)
    model_metrics_dict['max_criteria_metric_lst'] = max_criteria_metric_lst

    model_metrics_str += '***** max criteria metric scores thresholds table ***** \n'
    model_metrics_str += (list_2_csv_string(max_criteria_metric_lst) + '\n')
    model_metrics_str += '\n\n'

    print max_criteria_metric_df

    print '\n----- metrics scores on 100 thresholds  ----'
    thresholds_metrics_100_dict, thresholds_metrics_100_lst, ks, ks_threshold = thresholds_metrics_100(metrics_obj)
    model_metrics_dict['thresholds_scores_df_s_lst'] = thresholds_metrics_100_lst
    model_metrics_dict['ks'] = ks
    model_metrics_dict['ks_threshold'] = ks_threshold

    model_metrics_str += '***** metrics scores on 100 thresholds ***** \n'
    model_metrics_str += (list_2_csv_string(thresholds_metrics_100_lst) + '\n')
    model_metrics_str += '***** ks *****\n ' + str(ks) + '\n'
    model_metrics_str += '***** ks threshold ****\n ' + str(ks_threshold) + '\n'
    model_metrics_str += '\n\n'

    print '\n----- confusion matrix at thresold 0.5 -----'
    confusion_matrix_dict = confusion_matrix(metrics_obj, threshold_lst=[i * 0.01 for i in range(100)],
                                             metrics_dict=thresholds_metrics_100_dict)
    thresholds_c_matrix_lst = list()
    for i in range(100):
        thresholds_c_matrix_lst.append(dict({'threshold': i * 0.01, 'value': confusion_matrix_dict[round(i * 0.01, 2)]}))
    model_metrics_dict['c_matrix_lst'] = thresholds_c_matrix_lst

    model_metrics_str += '***** confusion matrix at thresold 0.5 ***** \n'
    print thresholds_c_matrix_lst[49]
    model_metrics_str += (list_2_csv_string(thresholds_c_matrix_lst[49]['value']) + '\n')
    model_metrics_str += '\n\n'

    print '\n----- metrics scores on 10 groups ----'
    score_group_threshold_10_dict = score_group_metrics_10(metrics_obj, thresholds_metrics_100_dict)

    score_group_threshold_10_lst = list()
    for threshold in score_group_threshold_10_dict.keys():
        threshold = round(threshold, 2)
        score_group_threshold_10_lst.append(dict({'threshold': threshold,
                                                  'value': score_group_threshold_10_dict[threshold]}))

    model_metrics_dict['score_group_threshold_10_dict'] = score_group_threshold_10_lst

    model_metrics_str += '***** metrics scores on 10 groups at threshold 0.5 ***** \n'

    print score_group_threshold_10_dict

    model_metrics_str += (list_2_csv_string(score_group_threshold_10_dict[0.5]) + '\n')
    model_metrics_str += '\n\n'

    print '\n-----  metrics scores with 10 topN threshold  ----'
    topn_metrics_10_lst = topn_metrics_10(metrics_obj)
    model_metrics_dict['topn_metrics_10_lst'] = topn_metrics_10_lst

    model_metrics_str += '***** metrics scores with 10 topN threshold ***** \n'
    model_metrics_str += (list_2_csv_string(topn_metrics_10_lst) + '\n')
    model_metrics_str += '\n\n'

    return model_metrics_dict, model_metrics_str, gains_lift_lst, thresholds_metrics_100_lst


def export_kmeans_model_propertyies_report(trained_model, xval, valid):
    """

    :return: 
    """
    print '\n--- k means model properties ---'
    model_properties_str = ''
    model_properties_dict = get_km_model_properties(trained_model)

    model_properties_str += '***** model properties ***** \n'
    for metric in model_properties_dict.keys():
        model_properties_str += (str(metric) + ': ' + str(model_properties_dict[metric]) + '\n')
    model_properties_str += '\n\n'

    print model_properties_dict

    print '\n--- k means model properties on all sets ---'
    model_metrics_dict = get_km_model_properties_by_set(trained_model, xval=xval, valid=valid)

    model_properties_str += '***** model properties on all sets ***** \n'
    for metric in model_metrics_dict.keys():
        model_properties_str += (str(metric) + ': ' + str(model_metrics_dict[metric]) + '\n')
    model_properties_str += '\n\n'

    print model_metrics_dict

    return model_properties_dict, model_properties_str, model_metrics_dict


def export_kmeans_model_metrics_report(metrics_obj):
    """

    :return: 
    """
    model_metrics_dict = get_km_test_metrics(metrics_obj=metrics_obj)
    model_metrics_str = ''
    model_metrics_str += '***** model test metrics ***** \n'
    for metric in model_metrics_dict.keys():
        model_metrics_str += (str(metric) + ': ' + str(model_metrics_dict[metric]) + '\n')
    model_metrics_str += '\n\n'

    return model_metrics_dict, model_metrics_str

