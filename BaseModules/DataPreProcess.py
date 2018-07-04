# -*- coding:utf-8 -*-

# The functions of this module include:
#    1. convert ia and label columns to factor;
#    2. calculate statistics of data set, numeric and factor separately;
#    3. calculate samples with too many missing values;
#    4. delete samples;
#    5. calculate missing value amount of each variable;
#    6. calculate std value of each variable;
#    7. calculate level amount of each variable;
#    8. delete variables;
#    9. random sampling(can be used for under-sampling);
#    10. fill missing value;
#    11. cut numerical variables into bins based on quantiles;
#    12. calculate woe ;
#    13. conduct Chi2 test for 2*2 table;
#    14. combine bins based on Chi2 test;
#    15. calculate iv value;
#

import h2o
import math
import copy


def process_id_label_type(data_df, id_name=None, target_name=None):
    """
    convert these two cols to factor
    :param data_df: a data frame
    :param id_name: id name
    :param target_name: label name
    :return: a processed data frame
    """
    if id_name:
        data_df[id_name] = data_df[id_name].ascharacter().asfactor()
    if target_name:
        data_df[target_name] = data_df[target_name].ascharacter().asfactor()
        data_df[target_name].levels()
    return data_df


def cal_numeric_stat(data_df, numeric_vars):
    """
    calculate statistics of numerical variables in data set
    :param data_df: a data frame
    :param numeric_vars: a list of numeric variables
    :return: a statistical dict of numerical variables
    """
    statics = data_df[numeric_vars].summary(return_data=True)
    medians = dict(zip(numeric_vars, data_df[numeric_vars].median(na_rm=True)))

    statics_dict = dict()
    statics_list = list([['variable', 'max', 'min', 'mean', 'sigma', 'median', 'missing_count']])
    for key in statics:
        tmp_dict = dict()
        tmp_list = list([key])

        if isinstance(statics[key]['maxs'], list) and len(statics[key]['maxs']) > 0:
            tmp_dict['max'] = statics[key]['maxs'][0]
            tmp_list.append(statics[key]['maxs'][0])
        else:
            tmp_dict['max'] = None
            tmp_list.append(None)

        if isinstance(statics[key]['mins'], list) and len(statics[key]['mins']) > 0:
            tmp_dict['min'] = statics[key]['mins'][0]
            tmp_list.append(statics[key]['mins'][0])
        else:
            tmp_dict['min'] = None
            tmp_list.append(None)

        tmp_dict['mean'] = statics[key]['mean']
        tmp_list.append(statics[key]['mean'])

        tmp_dict['sigma'] = statics[key]['sigma']
        tmp_list.append(statics[key]['sigma'])

        tmp_dict['median'] = medians[key]
        tmp_list.append(medians[key])

        tmp_dict['missing_count'] = statics[key]['missing_count']
        tmp_list.append(statics[key]['missing_count'])

        statics_dict[key] = tmp_dict
        statics_list.append(tmp_list)

    return statics_dict, statics_list


def cal_factor_stat(data_df, factor_vars):
    """
    calculate statistics of factor variables in data set
    :param data_df: a data frame 
    :param factor_vars: a list of factor variables
    :return: a statistical dict of factor variables
    """
    statics = data_df[factor_vars].summary(return_data=True)
    statics_dict = dict()
    statics_list = list([['variable', 'level_num', 'missing_count', 'most_freq_level', 'levels']])
    for key in statics:
        tmp_dict = dict()
        tmp_list = list([key])

        tmp_dict['levels'] = statics[key]['domain']
        if isinstance(tmp_dict['levels'], list):
            tmp_list.append(len(statics[key]['domain']))
        else:
            tmp_list.append(None)

        tmp_dict['missing_count'] = statics[key]['missing_count']
        tmp_list.append(statics[key]['missing_count'])

        # groupby_lst = h2o.as_list(data_df[factor_vars].group_by(by=key).count().frame, use_pandas=False, header=False)
        tmp_dict['levels_amount'] = cal_vars_levels_amount(data_df[factor_vars], key)

        max_amount = 0
        most_level = ''
        for level in tmp_dict['levels_amount'].keys():
            tmp_dict['levels_amount'][level] = int(tmp_dict['levels_amount'][level])
            if tmp_dict['levels_amount'][level] > max_amount:
                most_level = level
                max_amount = tmp_dict['levels_amount'][level]
        tmp_dict['most_freq_level'] = most_level
        tmp_list.append(most_level)
        if isinstance(statics[key]['domain'], list):
            tmp_list.append('|'.join(statics[key]['domain']))
        else:
            tmp_list.append(None)

        statics_dict[key] = tmp_dict
        statics_list.append(tmp_list)

    return statics_dict, statics_list


def cal_vars_levels_amount(data_df, factor_var):
    """
    
    :return: 
    """
    groupby_lst = h2o.as_list(data_df.group_by(by=factor_var).count().frame, use_pandas=False, header=False)
    return dict(groupby_lst)


def label_static(label_col):
    """
    calculate amount of each levels in a column
    :param label_col: a single label column of the data frame 
    :return: a count dict
    """
    label_col = label_col.asfactor()
    count_dict = dict(h2o.as_list(label_col.group_by(by=label_col.names[0]).count().get_frame(),
                                  use_pandas=False, header=False))
    for key in count_dict.keys():
        if count_dict[key] is not None:
            count_dict[key] = int(count_dict[key])
    return count_dict


def cal_samples_miss(data_df):
    """
    extract col numbers of missing value of each sample 
    :param data_df: a data frame
    :return: a list with missing value amount of each row (ordered by index)
    """
    row_miss_lst = [int(i[0]) for i in
                    h2o.as_list(data_df.isna().sum(axis=1, return_frame=True), use_pandas=False, header=False)]

    return row_miss_lst


def del_samples(data_df, remove_index_lst=None):
    """
    delete samples by index
    :param data_df: a data frame
    :param remove_index_lst: a list of row index that to be remove from data frame 
    :return: a data frame after removing pointed rows
    """
    if len(remove_index_lst) > 0:
        data_df = data_df.drop(remove_index_lst, axis=0)

    return data_df


def cal_vars_miss(data_df):
    """
    calculate amount of missing values of each variables
    :param data_df: a data frame
    :return: a calculated dict 
    """
    col_miss_dict = dict(zip(data_df.names, data_df.nacnt()))

    return col_miss_dict


def cal_vars_std(data_df):
    """
    calculate variance of each variable
    :param data_df: a data frame
    :return: a calculated dict 
    """
    col_std_dict = dict(zip(data_df.names, data_df.sd(na_rm=True)))

    return col_std_dict


def cal_vars_levels(data_df):
    """
    collect levels of each variable
    :param data_df: a data frame
    :return: a calculated dict 
    """
    levels_dict = dict(zip(data_df.names, data_df.levels()))

    return levels_dict


def del_vars(data_df, remove_var_lst):
    """
    delete variables
    :param data_df: a data frame
    :param remove_var_lst: a list of variable to be removed from the data frame
    :return: a data frame after removing pointed variables
    """
    if len(remove_var_lst) > 0:
        data_df = data_df.drop(remove_var_lst[:], axis=1)

    return data_df


def random_sampling(data_df, ratio=None, sub_num=None):
    """
    random sampling by ratio or sample number
    :param data_df: a data frame
    :param ratio: sampling ratio
    :param sub_num: sampling number
    :return: a sampled data frame
    """
    if ratio is None and sub_num is not None:
        ratio = 1.0*sub_num / data_df.shape[0]
    assert isinstance(ratio, float) and 0 < ratio < 1, \
        'sample ratio or number is out of range !'

    names = data_df.names
    sampled_df = data_df.split_frame(ratios=[ratio])[0]
    sampled_df = sampled_df[names]

    return sampled_df


def fill_missing(data_df, fill_dict):
    """
    fill missing value
    :param data_df: a data frame
    :param fill_dict: a dict that give a value for each variable to fill
    :return: a filled data frame
    """
    if isinstance(fill_dict, dict) and len(fill_dict.keys()) > 0:
        for key in fill_dict.keys():
            data_df[data_df[key].isna(), key] = fill_dict[key]

    return data_df


def numeric_quantile_bin(data_df, cal_numeric_cols, nbin=20):
    """
    cut numerical variables into buckets by quantiles
    :param data_df: a data frame
    :param cal_numeric_cols: numerical columns to be cut
    :param nbin: bucket number
    :return: a data frame after cutting, and a dict with cutting info
    """
    percentiles = [i * 1.0 / nbin for i in range(nbin + 1)]
    numeric_bin_dict = dict()
    for col in cal_numeric_cols:
        break_lst = h2o.as_list(data_df[col].quantile(prob=percentiles, combine_method=u'interpolate')[:, 1],
                                use_pandas=False, header=False)
        break_lst = [float(i[0]) for i in break_lst]
        break_labels = [col + '_' + str(i + 1) for i in range(nbin)]
        data_df[col] = data_df[col].cut(break_lst, labels=break_labels, include_lowest=True, right=True, dig_lab=3)
        numeric_bin_dict[col] = [break_labels, break_lst]

    return data_df, numeric_bin_dict


def cal_woe(data_df, cal_cols, target_name):
    """
    calculate woe
    :param data_df: a data frame
    :param cal_cols: a list of variables to calculate woe
    :param target_name: label name
    :return: a dict with woe info of each variable
    """
    labels = data_df[target_name].categories()
    data_df[target_name] = data_df[target_name].ascharacter()

    if set(labels) != set(['1', '0']):
        label_map = dict(zip(labels, ['1', '0']))
        data_df[data_df[target_name] == labels[0], target_name] = label_map[labels[0]]
        data_df[data_df[target_name] == labels[1], target_name] = label_map[labels[1]]

    data_df[target_name] = data_df[target_name].asnumeric()

    pos_sum = data_df[target_name].sum()
    neg_sum = data_df[target_name].shape[0] - pos_sum

    woe_dict = dict()
    for col in cal_cols:
        tmp_groupby_df = data_df[[col, target_name]]\
            .group_by(by=col)\
            .count()\
            .sum(col=target_name)\
            .frame
        print tmp_groupby_df
        tmp_lst = h2o.as_list(tmp_groupby_df, use_pandas=False, header=False)
        headers = tmp_groupby_df.names
        header_index_map = dict({col: headers.index(col),
                                 'nrow': headers.index('nrow'),
                                 'sum': headers.index('sum_' + target_name)})

        col_woe_dict = dict()
        for statics in tmp_lst:
            level = statics[header_index_map[col]]
            amount = float(statics[header_index_map['nrow']])
            pos_amount = float(statics[header_index_map['sum']])
            neg_amount = amount - pos_amount
            tmp_dict = dict({'count': amount, 'pos': pos_amount, 'neg': neg_amount,
                             'pct_1': pos_amount * 1.0 / pos_sum, 'pct_0': neg_amount * 1.0 / neg_sum})
            correction_factor = 1.0 * (tmp_dict['pct_1'] + tmp_dict['pct_0']) / 200
            correction = (tmp_dict['pct_1'] == 0 or tmp_dict['pct_0'] == 0)
            tmp_dict['woe'] = math.log((tmp_dict['pct_1'] + correction * correction_factor)
                                       / (tmp_dict['pct_0'] + correction * correction_factor))
            col_woe_dict[level] = tmp_dict

        woe_dict[col] = col_woe_dict

    return woe_dict


def cal_woe_df(data_df, cal_col, target_name):
    """
    calculate woe
    :param data_df: a data frame
    :param cal_col: a variables to calculate woe
    :param target_name: label name
    :return: a dict with woe info of each variable
    """
    labels = data_df[target_name].categories()
    data_df[target_name] = data_df[target_name].ascharacter()

    if set(labels) != set(['1', '0']):
        label_map = dict(zip(labels, ['1', '0']))
        data_df[data_df[target_name] == labels[0], target_name] = label_map[labels[0]]
        data_df[data_df[target_name] == labels[1], target_name] = label_map[labels[1]]

    data_df[target_name] = data_df[target_name].asnumeric()

    pos_sum = data_df[target_name].sum()
    neg_sum = data_df[target_name].shape[0] - pos_sum
    print '\n--- mark10 ---'
    tmp_groupby_df = data_df[[cal_col, target_name]]\
        .group_by(by=cal_col)\
        .count()\
        .sum(col=target_name)\
        .frame
    print '\n--- mark11 ---'
    tmp_groupby_df['neg_amount'] = tmp_groupby_df['nrow'] - tmp_groupby_df['sum_' + target_name]
    correction_factor = 0.01
    tmp_groupby_df['woe'] = ((tmp_groupby_df['sum_' + target_name] * 1.0 / pos_sum + correction_factor)
                             / (tmp_groupby_df['neg_amount'] * 1.0 / neg_sum + correction_factor)).log()

    return tmp_groupby_df


def chi2_test_2x2(data_lst):
    """
    conduct a 2*2 Chi-square test
    :param data_lst: a list of two lists like [[a, b], [c, d]]
    :return: a Chi2 value(float) and its p value region(a two value list)
    """
    # chi2 value - p table for 1 degree
    chi2_table_1df = {0.995: 0.0000393, 0.975: 0.000982, 0.20: 1.642, 0.10: 2.706, 0.05: 3.841,
                      0.025: 5.024, 0.02: 5.412, 0.01: 6.635, 0.005: 7.879, 0.002: 9.550, 0.001: 10.828}
    a = data_lst[0][0]
    b = data_lst[0][1]
    c = data_lst[1][0]
    d = data_lst[1][1]
    abcd_sum = a + b + c + d
    if len(filter(lambda x: x == 0, [a, b, c, d])) == 1:
        # it means two sample are obviously different
        chi2_value = 100
    elif a > 0 and b == 0 and c > 0 and d == 0:
        # it means two sample are similar
        chi2_value = 0
    elif a == 0 and b > 0 and c == 0 and d > 0:
        # it means two sample are similar
        chi2_value = 0
    elif len(filter(lambda x: x == 0, [a, b, c, d])) in [3, 2]:
        # it means two sample are obviously different
        chi2_value = 100
    elif len(filter(lambda x: x == 0, [a, b, c, d])) == 4:
        # it means two sample are similar
        chi2_value = 0
    elif a < 5 or b < 5 or c < 5 or d < 5:
        # correction chi2 test
        correction_factor = 0.1
        chi2_value = (1.0 * (abs(a * d - b * c) - 1.0 * abcd_sum / 2)**2 * abcd_sum + correction_factor) \
                     / ((a + b) * (c + d) * (a + c) * (b + d) + correction_factor)
    else:
        # normal chi2 test
        correction_factor = 0.1
        chi2_value = (1.0 * (a * d - b * c)**2 * abcd_sum + correction_factor) \
                     / ((a + b) * (c + d) * (a + c) * (b + d) + correction_factor)

    p_value = [0.0, 1.0]
    for alpha in sorted(list(chi2_table_1df.keys())):
        if chi2_value > chi2_table_1df[alpha]:
            p_value[1] = alpha
            break
        elif chi2_value == chi2_table_1df[alpha]:
            p_value[1] = alpha
            p_value[0] = alpha
            break
        elif chi2_value < chi2_table_1df[alpha]:
            p_value[0] = alpha

    return chi2_value, p_value


def factor_woe_order(woe_dict):
    """
    order levels of factor variable by woe value
    :param woe_dict: a dict with woe value of each level of factor variables
    :return: a dict with factor variables as its key, and a list as its value (it include two lists, 
    a list of ordered levels, and a list of level's corresponding order)
    """
    factor_level_dict = dict()
    for col in woe_dict.keys():
        col_woe_dict = woe_dict[col]
        tmp_dict = dict([(label, col_woe_dict[label]['woe']) for label in col_woe_dict.keys()])
        tmp_lst = sorted(tmp_dict.items(), key=lambda d: d[1])
        factor_level_dict[col] = [[i[0] for i in tmp_lst], [i + 1 for i in range(len(tmp_lst))]]

    return factor_level_dict


def bins_merge(woe_dict, bin_dict, p_threshold=0.1):
    """
    combine bins based on Chi2 test between adjacent bins
    :param woe_dict: a dict with woe info of each variable
    :param bin_dict；a dict with bin info of each variable
    :param p_threshold: the alpha value to decision whether combine two bins
    :return: a dict with woe info and a dict with bin info after combination
    """

    for col in woe_dict.keys():
        col_woe_dict = woe_dict[col]
        bin_labels = bin_dict[col][0]
        bin_breaks = bin_dict[col][1]
        exist_labels = list(col_woe_dict.keys())
        for i in range(len(bin_labels)):
            if bin_labels[i] not in exist_labels:
                bin_labels[i] = None
                bin_breaks[i + 1] = None
        bin_labels = filter(lambda x: x is not None, bin_labels)
        bin_breaks = filter(lambda x: x is not None, bin_breaks)

        for i in range(len(bin_labels) - 1):
            label_1 = bin_labels[i]
            data_lst = list()
            data_lst.append([1.0 * col_woe_dict[label_1]['pos'],
                             1.0 * col_woe_dict[label_1]['neg']])
            label_2 = bin_labels[i + 1]
            data_lst.append([1.0 * col_woe_dict[label_2]['pos'],
                             1.0 * col_woe_dict[label_2]['neg']])
            # calculate chi2 value
            chi2_value, p_value = chi2_test_2x2(data_lst)

            if p_value[0] >= p_threshold:
                # combine bin info of these two bins
                label_1_dict = col_woe_dict[label_1]
                label_2_dict = col_woe_dict[label_2]
                tmp_dict = dict({'count': label_1_dict['count'] + label_2_dict['count'],
                                 'pos': label_1_dict['pos'] + label_2_dict['pos'],
                                 'neg': label_1_dict['neg'] + label_2_dict['neg'],
                                 'pct_1': label_1_dict['pct_1'] + label_2_dict['pct_1'],
                                 'pct_0': label_1_dict['pct_0'] + label_2_dict['pct_0']})

                # correction for avoiding 0 division and log(0)
                correction_factor = 1.0 * (tmp_dict['pct_1'] + tmp_dict['pct_0']) / 200
                correction = (tmp_dict['pct_1'] == 0 or tmp_dict['pct_0'] == 0)
                tmp_dict['woe'] = math.log((tmp_dict['pct_1'] + correction * correction_factor)
                                           / (tmp_dict['pct_0'] + correction * correction_factor))

                # updata woe info dict
                del col_woe_dict[label_2]
                del col_woe_dict[label_1]
                bin_labels[i + 1] = label_1 + '_' + label_2
                col_woe_dict[bin_labels[i + 1]] = tmp_dict

                bin_labels[i] = None
                bin_breaks[i + 1] = None

        # updata
        woe_dict[col] = col_woe_dict
        bin_dict[col][0] = bin_labels
        bin_dict[col][1] = bin_breaks

    return woe_dict, bin_dict


def cal_iv(woe_dict):
    """
    calculate iv value
    :param woe_dict: a dict of woe values of each variable
    :return: a dict with iv values of each variable
    """
    iv_dict = dict()
    iv_list = list([['variable', 'iv']])
    woe_list = list([['variable', 'bin', 'woe']])
    for col in woe_dict.keys():
        iv = 0
        tmp_dict = woe_dict[col]
        for level in tmp_dict.keys():
            woe_list.append([col, level, tmp_dict[level]['woe']])
            iv += (tmp_dict[level]['pct_1'] - tmp_dict[level]['pct_0']) * tmp_dict[level]['woe']
        iv_dict[col] = iv
        iv_list.append([col, iv])
    return iv_dict, iv_list, woe_list


def transfer_levels2woe(data_df, cal_cols, target_col):
    """
    
    :return: 
    """
    print data_df.types
    map_dict = dict()
    for cal_col in cal_cols:
        print '\n--- ' + cal_col + ' ---'
        # data_df[target_col] = data_df[target_col].asfactor()
        # data_df[data_df[cal_col].isna(), cal_col] = 'NoneValue'
        print '\n--- none ---'
        print data_df[data_df[cal_col].isna(), cal_col]
        tmp_df = copy.copy(data_df[[cal_col, target_col]])

        factors_woe_df = cal_woe_df(tmp_df, cal_col=cal_col, target_name=target_col)
        factors_woe_df = factors_woe_df[[cal_col, 'woe']]
        print '\n--- mark1 ---'
        factors_woe_df['woe'] = (factors_woe_df['woe'].scale() * 10).round().ascharacter().asfactor()
        factors_woe_df[cal_col] = factors_woe_df[cal_col].asfactor()
        print '\n--- mark2 ---'
        data_df[cal_col] = data_df[cal_col].asfactor()
        data_df = data_df.merge(factors_woe_df, all_x=True, by_x=[cal_col], by_y=[cal_col])
        print '\n--- mark3 ---'
        print factors_woe_df
        map_dict[cal_col] = h2o.as_list(factors_woe_df, use_pandas=False, header=True)
        #print map_dict
        #del factors_woe_df
        #print data_df
        #data_df[cal_col] = data_df['woe']
        data_df = data_df.drop([cal_col])
        #print data_df
        data_df.set_name(col='woe', name=cal_col)
        print data_df

    return data_df, map_dict

