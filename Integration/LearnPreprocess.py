from BaseModules.DataPreProcess import *
import json
import copy



def learn_cal_summary_statics(data_obj, learn_conf_dict):
    """
    
    :return: 
    """

    if learn_conf_dict['preprocess_conf']['cal_statics_sampling']:
        sample_ratio = learn_conf_dict['preprocess_conf']['cal_statics_sampling']
        nrows, ncols = data_obj['df'].shape
        if isinstance(sample_ratio, int):
            assert sample_ratio <= nrows, 'sampling amount is set bigger than total set!'
        elif isinstance(sample_ratio, float):
            assert 0 < sample_ratio < 1, 'sampling ratio out of range!'
        else:
            if nrows > 10000:
                if ncols > 500:
                    learn_conf_dict['preprocess_conf']['cal_statics_sampling'] = 10000
                else:
                    learn_conf_dict['preprocess_conf']['cal_statics_sampling'] = 20000
            else:
                learn_conf_dict['preprocess_conf']['cal_statics_sampling'] = None

    numeric_statics_list = list()
    factor_statics_list = list()
    if learn_conf_dict['preprocess_conf']['cal_statics_universal'] or \
            learn_conf_dict['preprocess_conf']['cal_statics_sampling']:

        sampling_num = learn_conf_dict['preprocess_conf']['cal_statics_sampling']
        data_df = data_obj['df']
        factor_vars = data_obj['info']['factor_vars']
        numeric_vars = data_obj['info']['numeric_vars']
        variables = data_obj['info']['variables']
        if sampling_num:
            if sampling_num > 1:
                data_sampled = random_sampling(data_df, sub_num=sampling_num)
            else:
                data_sampled = random_sampling(data_df, ratio=sampling_num)
        else:
            data_sampled = data_df

        print '\n--- calculate factor variable statistics ---'
        factor_statics_dict, factor_statics_list = cal_factor_stat(data_sampled[:, variables], factor_vars)
        print json.dumps(factor_statics_dict, indent=1)

        print '\n--- calculate numeric variable statistics ---'
        numeric_statics_dict, numeric_statics_list = cal_numeric_stat(data_sampled[:, variables], numeric_vars)
        print json.dumps(numeric_statics_dict, indent=1)

        # save statics
        data_obj['info']['factor_statics'] = factor_statics_dict
        data_obj['info']['numeric_statics'] = numeric_statics_dict

    return data_obj, numeric_statics_list, factor_statics_list


def learn_cal_samples_miss(data_obj):
    """
    
    :return: 
    """
    print '\n--- calculate missing variable number of each sample ---'
    variables = data_obj['info']['variables']
    row_miss_lst = cal_samples_miss(data_obj['df'][:, variables])
    data_obj['info']['row_miss_lst'] = row_miss_lst
    # print row_miss_lst
    return data_obj


def learn_cal_vars_levels(data_obj, learn_conf_dict):
    """
    
    :return: 
    """
    print '\n--- calculate level numbers of factor variables ---'
    levels_dict = dict()
    if learn_conf_dict['preprocess_conf']['cal_statics_universal']:
        if data_obj['info']['factor_statics'].keys() is not None \
                and len(data_obj['info']['factor_statics'].keys()) > 0:
            for var in data_obj['info']['factor_statics'].keys():
                levels_dict[var] = data_obj['info']['factor_statics'][var]['levels']
    else:
        factor_vars = data_obj['info']['factor_vars']
        if factor_vars is not None and len(factor_vars) > 0:
            levels_dict = cal_vars_levels(data_obj['df'][:, factor_vars])

    data_obj['info']['levels_dict'] = levels_dict

    return data_obj


def learn_cal_vars_miss(data_obj, learn_conf_dict):
    """
    
    :return: 
    """
    print '\n--- calculate missing sample number of each variable ---'
    if learn_conf_dict['preprocess_conf']['cal_statics_universal']:
        col_miss_dict = dict()
        for var in data_obj['info']['numeric_statics'].keys():
            col_miss_dict[var] = data_obj['info']['numeric_statics'][var]['missing_count']
        for var in data_obj['info']['factor_statics'].keys():
            col_miss_dict[var] = data_obj['info']['factor_statics'][var]['missing_count']
    else:
        variables = data_obj['info']['variables']
        col_miss_dict = cal_vars_miss(data_obj['df'][:, variables])
    data_obj['info']['col_miss_dict'] = col_miss_dict

    # print json.dumps(col_miss_dict, indent=1)
    return data_obj


def learn_cal_vars_std(data_obj, learn_conf_dict):
    """
    
    :return: 
    """
    print '\n--- calculate standard deviation of numeric variable ---'
    col_std_dict = dict()
    if learn_conf_dict['preprocess_conf']['cal_statics_universal']:
        if data_obj['info']['numeric_statics'].keys() is not None and \
                        len(data_obj['info']['numeric_statics'].keys()) > 0:
            for var in data_obj['info']['numeric_statics'].keys():
                col_std_dict[var] = data_obj['info']['numeric_statics'][var]['sigma']
    else:
        numeric_vars = data_obj['info']['numeric_vars']
        if numeric_vars is not None and len(numeric_vars) > 0:
            col_std_dict = cal_vars_std(data_obj['df'][:, numeric_vars])
    data_obj['info']['col_std_dict'] = col_std_dict
    # print json.dumps(col_std_dict, indent=1)
    return data_obj


def learn_del_samples(data_obj, learn_conf_dict):
    """

    :return: 
    """
    print '\n--- delete samples ---'
    remove_indexs = []
    del_threshold_ratio = learn_conf_dict['preprocess_conf']['max_sample_miss_prop']
    variables = data_obj['info']['variables']
    del_threshold_num = int(len(variables) * del_threshold_ratio)
    row_miss_lst = data_obj['info']['row_miss_lst']
    for i in range(len(row_miss_lst)):
        if row_miss_lst[i] > del_threshold_num:
            remove_indexs.append(i)
    data_obj['info']['remove_indexs'] = remove_indexs
    data_obj['df'] = del_samples(data_obj['df'], remove_index_lst=remove_indexs)
    print data_obj['df']
    return data_obj


def learn_del_vars(data_obj, learn_conf_dict, nrows):
    """
    
    :return: 
    """
    print '\n--- test delete variables ---'
    remove_vars = list()

    # # based on factor levels amount
    factor_levels_dict = data_obj['info']['levels_dict']
    del_threshold_ratio = learn_conf_dict['preprocess_conf']['max_factor_levels_prop']
    del_threshold_num = int(nrows * del_threshold_ratio) if del_threshold_ratio <= 1 else del_threshold_ratio
    for var in factor_levels_dict.keys():
        if isinstance(factor_levels_dict[var], list) and len(factor_levels_dict[var]) > del_threshold_num:
            remove_vars.append(var)
        if isinstance(factor_levels_dict[var], list) and len(factor_levels_dict[var]) == 1 and data_obj['info']['col_miss_dict'][var] == 0:
            remove_vars.append(var)

    # # based on missing amount
    factor_miss_dict = data_obj['info']['col_miss_dict']
    del_threshold_ratio = learn_conf_dict['preprocess_conf']['max_variable_miss_prop']
    del_threshold_num = int(nrows * del_threshold_ratio)
    print 'var del_threshold_num: ' + str(del_threshold_num)
    for var in factor_miss_dict.keys():
        if factor_miss_dict[var] >= del_threshold_num:
            remove_vars.append(var)

    # # based on numeric std
    numeric_std_dict = data_obj['info']['col_std_dict']
    for var in numeric_std_dict.keys():
        if numeric_std_dict[var] == 0 and data_obj['info']['col_miss_dict'][var] == 0:
            remove_vars.append(var)

    remove_vars = list(set(remove_vars))
    for var in remove_vars:
        data_obj['info']['variables'].remove(var)
        if var in data_obj['info']['numeric_vars']:
            data_obj['info']['numeric_vars'].remove(var)
        if var in data_obj['info']['factor_vars']:
            data_obj['info']['factor_vars'].remove(var)

    data_obj['info']['remove_vars'] = remove_vars

    for var in remove_vars:
        del data_obj['schema'][var]
    data_obj['df'] = del_vars(data_obj['df'], remove_var_lst=remove_vars)

    print data_obj['df']
    return data_obj


def learn_under_sampling(data_obj, learn_conf_dict):
    """
    
    :return: 
    """
    print '\n--- under sampling  ---'
    data_df = data_obj['df']
    target_name = data_obj['info']['target_name']
    label_count_dict = label_static(data_df[target_name])
    print json.dumps(label_count_dict, indent=1)

    minor_amount, major_amount = sorted(list(label_count_dict.values()), reverse=False) # increse
    if major_amount * 1.0 / minor_amount > learn_conf_dict['preprocess_conf']['unbalanced_cutoff']:
        sample_num = int(minor_amount * learn_conf_dict['preprocess_conf']['unbalanced_cutoff'])
        minor_label = ''
        major_label = ''
        for label in label_count_dict.keys():
            if label_count_dict[label] == minor_amount:
                minor_label = label
            else:
                major_label = label
        print '\n--- original row number: ' + str(data_df.shape[0])

        # ori_names  =data_df.names
        # data_df['index_tag'] = 1
        # cum_1 = data_df['index_tag'].cumsum()
        # cum_1.set_name(col=cum_1.names[0], name='index_tag')
        # data_df = data_df[ori_names].cbind(cum_1)
        # minor_index = data_df[data_df[target_name] == minor_label]['index_tag']
        # major_index = data_df[data_df[target_name] == major_label]['index_tag']
        # major_index = random_sampling(major_index, sub_num=sample_num)
        # sample_index = minor_index.rbind(major_index)
        # gener_lst = [[0] for i in range(data_df.shape[0] - sample_index.shape[0])]
        # sample_index = sample_index.rbind(h2o.H2OFrame(gener_lst, column_names=['index_tag']))

        # print sample_index
        # print data_df
        # data_df.drop(['index_tag'])
        # data_df = data_df.cbind(sample_index)
        # data_df = data_df[data_df['index_tag'] != 0]
        # data_df.drop(['index_tag'])

        ori_names  =data_df.names
        data_df['index_tag'] = 1
        cum_1 = data_df['index_tag'].cumsum()
        cum_1.set_name(col=cum_1.names[0], name='index_tag')
        data_df = data_df[ori_names].cbind(cum_1)
        minor_index = data_df[data_df[target_name] == minor_label]['index_tag']
        major_index = data_df[data_df[target_name] == major_label]['index_tag']
        major_index = random_sampling(major_index, sub_num=sample_num)
        sample_index = minor_index.rbind(major_index).sort(by='index_tag', ascending=[True])
        sample_index_lst = [int(i[0]) for i in h2o.as_list(sample_index - 1, use_pandas=False, header=False)]
        data_df = data_df[sample_index_lst, :]

        print data_df
        print data_df[target_name].table()      

        # minor_df = data_df[data_df[target_name] == minor_label]
        # for var in minor_df.names:
        #     minor_df[var] = minor_df[var].ascharacter()
        # major_df = random_sampling(data_df[data_df[target_name] == major_label], sub_num=sample_num)
        # for var in major_df.names:
        #     major_df[var] = major_df[var].ascharacter()

        # data_df = minor_df.rbind(major_df)
        # type_dict = data_obj['schema']
        # for var in data_df.names:
        #     if type_dict[var] == 'numeric':
        #         data_df[var] = data_df[var].asnumeric()
        #     else:
        #         data_df[var] = data_df[var].asfactor()

        print data_df
        print '\n---sampled row number in proportion: ' + str(data_df.shape[0])

    data_obj['df'] = data_df
    return data_obj


def learn_fill_missing_value(data_obj, learn_conf_dict):
    """
    
    :return: 
    """
    print '\n--- fill missing values ---'
    if learn_conf_dict['train_conf']['algorithm'] == 'NB':
        tmp_fill_dict = dict()
        factor_vars = data_obj['info']['factor_vars']
        numeric_vars = data_obj['info']['numeric_vars']
        if learn_conf_dict['preprocess_conf']['cal_statics_universal']:

            tmp_dict = data_obj['info']['factor_statics']
            for var in factor_vars:
                tmp_fill_dict[var] = tmp_dict[var]['most_freq_level']

            tmp_dict = data_obj['info']['numeric_statics']
            for var in numeric_vars:
                tmp_fill_dict[var] = tmp_dict[var]['median']

            learn_conf_dict['preprocess_conf']['fill_dict'] = tmp_fill_dict
        else:
            if factor_vars is not None and len(factor_vars) > 0:
                for var in factor_vars:
                    levels_dict = cal_vars_levels_amount(data_obj['df'], var)
                    max_level = ''
                    max_freq = 0
                    for level in levels_dict.keys():
                        if levels_dict[level] > max_freq:
                            max_freq = levels_dict[level]
                            max_level = level
                    tmp_fill_dict[var] = max_level

            if numeric_vars is not None and len(numeric_vars) > 0:
                tmp_dict = dict(zip(numeric_vars, data_obj['df'][numeric_vars].median(na_rm=True)))
                for var in numeric_vars:
                    tmp_fill_dict[var] = tmp_dict[var]

        if isinstance(learn_conf_dict['preprocess_conf']['fill_dict'], dict):
            for var in tmp_fill_dict.keys():
                tmp_value = learn_conf_dict['preprocess_conf']['fill_dict'].get(var)
                if tmp_value:
                    tmp_fill_dict[var] = tmp_value
        learn_conf_dict['preprocess_conf']['fill_dict'] = tmp_fill_dict

    data_obj['df'] = fill_missing(data_obj['df'], learn_conf_dict['preprocess_conf']['fill_dict'])
    return data_obj


def learn_cal_iv(data_obj, learn_conf_dict):
    """
    
    :return: 
    """
    iv_list = list()
    woe_list = list()
    if learn_conf_dict['preprocess_conf']['cal_iv']:

        data_df = data_obj['df']
        factor_vars = data_obj['info']['factor_vars']
        numeric_vars = data_obj['info']['numeric_vars']
        target_name = data_obj['info']['target_name']

        if learn_conf_dict['preprocess_conf']['cal_statics_universal'] or \
                learn_conf_dict['preprocess_conf']['cal_statics_sampling']:
            fill_dict = dict()
            for var in numeric_vars:
                fill_dict[var] = data_obj['info']['numeric_statics'][var]['median']
            for var in factor_vars:
                fill_dict[var] = data_obj['info']['factor_statics'][var]['levels_amount']

            if learn_conf_dict['preprocess_conf']['fill_dict'] is not None:
                for var in learn_conf_dict['preprocess_conf']['fill_dict'].keys():
                    fill_dict[var] = learn_conf_dict['preprocess_conf']['fill_dict'][var]

        else:
            fill_dict = dict()
            if learn_conf_dict['preprocess_conf']['fill_dict'] is not None:
                for var in learn_conf_dict['preprocess_conf']['fill_dict'].keys():
                    fill_dict[var] = learn_conf_dict['preprocess_conf']['fill_dict'][var]
                vars_cal_median = [var for var in numeric_vars if var not in fill_dict.keys()]
                fill_dict.update(dict(zip(vars_cal_median, data_df[vars_cal_median].median(na_rm=True))))

        print '\n--- cut numeric variable into bins by quantiles ---'

        # make shallow copy
        df_woe = copy.copy(data_df)
        bin_df, numeric_bin_dict = numeric_quantile_bin(df_woe, numeric_vars)
        print bin_df
        # print json.dumps(numeric_bin_dict, indent=1)

        print '\n--- calculate woe ---'
        woe_dict = cal_woe(bin_df, numeric_vars + factor_vars, target_name)
        # print json.dumps(woe_dict, indent=1)

        print '\n--- order factor levels by woe ---'
        woe_factor_dict = dict()
        for i in factor_vars:
            woe_factor_dict[i] = woe_dict[i]
        factor_level_dict = factor_woe_order(woe_factor_dict)
        # print json.dumps(factor_level_dict, indent=1)

        print '\n--- merge bins by chi2 test for woe ---'
        bin_dict = dict(numeric_bin_dict.items() + factor_level_dict.items())
        woe_dict, bin_dict = bins_merge(woe_dict, bin_dict)
        # print json.dumps(woe_dict, indent=1)
        # print json.dumps(bin_dict, indent=1)

        print '\n--- calculate iv ---'
        iv_dict, iv_list, woe_list = cal_iv(woe_dict)
        data_obj['info']['iv_dict'] = iv_dict
        print json.dumps(iv_dict, indent=1)

    return data_obj, iv_list, woe_list


def learn_reduce_levels(data_obj, learn_conf_dict):
    """
    
    :param data_obj: 
    :param learn_conf_dict: 
    :return: 
    """
    print '\n--- reduce levels amount ---'
    from datetime import datetime
    start_time = datetime.now()
    print start_time
    data_df = data_obj['df']
    target_name = data_obj['info']['target_name']
    factor_vars = data_obj['info']['factor_vars']
    levels_dict = data_obj['info']['levels_dict']
    max_levels_amount = learn_conf_dict['preprocess_conf']['max_levels_amount']

    transfer_cols = list()
    for var in factor_vars:
        if len(levels_dict[var]) > max_levels_amount:
            transfer_cols.append(var)
    print transfer_cols
    print '\n'
    if len(transfer_cols) > 0:
        data_obj['df'], map_dict = transfer_levels2woe(data_df[:100000, :], transfer_cols, target_name)
        data_obj['info']['levels2woe_map'] = map_dict

    end_time = datetime.now()
    print end_time
    print 'running time: ' + str(end_time - start_time)

    return data_obj
