# -*- coding:utf-8 -*-

# The functions of this module include:
#    1. split dataframe by row;
#    2. merge dataframe by col;
#    3. union dataframe by row;
#

import h2o

def split_data(data_df, ratio, stratified=False, stratified_col=None):
    """
    split one data frame into two by rows
    :param data_df: a data frame
    :param ratio: split ratio for the first data frame
    :param stratified: whether stratify
    :param stratified_col: if stratify, assign a key column
    :return: a list of two data frame
    """
    assert ratio < 1, 'split ratio should less than 1 !'
    if stratified:
        stratsplit = data_df[str(stratified_col)].stratified_split(test_frac=(1 - ratio), seed=-1)
        df_lst = [data_df[stratsplit == "train"], data_df[stratsplit == "test"]]
    else:
        df_lst = data_df.split_frame(ratios=[ratio], destination_frames=None, seed=None)

    return df_lst


def merge_data(df_1, df_2, left_all=True, right_all=True, df_1_keys=None, df_2_keys=None):
    """
    merge two dataframe into one
    note: it dosen't like join function in sql,
        the row number of output frame is equal to left table if 'left_all=True'
        or right table if 'right_all=True'
    :param df_1: a data frame, left table
    :param df_2: a data frame , right table 
    :param left_all: if  include all rows from the left frame
    :param right_all:  include all rows from the right frame
    :param df_1_keys: key cols for merge on left table
    :param df_2_keys: key cols for merge on right table
    :return: merged data frame
    """
    assert len(df_1_keys) == len(df_2_keys), \
        'number of keys should be equal !'

    df_merged = df_1.merge(df_2,
                           all_x=left_all, all_y=right_all,
                           by_x=df_1_keys, by_y=df_2_keys)

    return df_merged

# h2o can't union type 'real' and type 'int', and type 'int' can't convert into 'real'
# this function can't be used now.
def union_data(df_1, df_2):
    """
    union two data frame into one 
    :param: df_1, df_1: two data frame to be unioned
    :return: a unioned dataframe
    """
    df_union = df_1.rbind(df_2)
    return df_union
