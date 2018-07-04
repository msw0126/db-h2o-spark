# -*- coding:utf-8 -*-
# The functions of this module include:
#    1. check id col;
#    2. check label col;
#    3. check type dict;
#    4. check col name and variable names in dict;
#    5. check data set volume;
#


def check_id(data_df, id_name):
    """
    check the dataframe: 
    1. if id col exist
    2. when id col exists, whether each row of the id col is unique
    3. if null value exists in id col
    :param: data_df: a data frame 
    :param: id_name: a specified id name
    :return: if id check passed
    """
    print '---- test debug ------------'
    print data_df.shape[0] 
    print data_df[str(id_name)].unique().shape
    print len(data_df[str(id_name)].unique())
    
    assert str(id_name) in data_df.col_names, \
        'a column with this id name is not in the dataframe'
    assert data_df.shape[0] == len(data_df[str(id_name)].unique()), \
        'there are same id between two rows in the dataframe'
    assert data_df[str(id_name)].nacnt()[0] == 0, \
        'there is null value in id col'
    return True


def check_label(data_df, target_name):
    """
    check the dataframe: 
    1. if label col exist
    2. if null value exists in label col
    :param: data_df: a data frame
    :param: target_name: a specified target name
    :return: if label check passed
    """
    assert str(target_name) in data_df.col_names, \
        'a column with this target name is not in the dataframe'
    assert data_df[str(target_name)].nacnt()[0] == 0, \
        'there is null value in label col'
    return True


def check_type_dict(type_lst, numeric_types=None, factor_types=None):
    """
    check type dictionary: 
    1. if it include two columns 
    2. the first line are 'variable' and 'type'
    3. if data types in dictionary are unexpected
    4. if there are duplicated variables
    :param: type_lst: a list of variables and types
    :param: numeric_types: a list of types that can be convert to numeric
    :param: factor_types: a list of types that can be convert to factor
    :return: a correct type list 
    """
    print numeric_types
    print factor_types

    assert len(type_lst[0]) == 2 and str(type_lst[0][0]) == 'variable' and str(type_lst[0][1]) == 'type', \
        ' the column names of type dict are wrong !'

    tmp_col_lst = []
    for i in type_lst:

        if i[0] != 'variable':

            assert len(i) == 2, \
                ' there are more or less than 2 cols in one row of the type dict csv, start with %s' % i[0]

            var_name = str(i[0])
            assert var_name not in tmp_col_lst, \
                'there are two %s variables in the dict' % var_name
            tmp_col_lst.append(str(i[0]))

            var_type = str(i[1])
            if var_type != 'numeric' and (isinstance(numeric_types, list) and (var_type.lower() in numeric_types)):
                i[1] = 'numeric'
            elif var_type != 'factor' and (isinstance(factor_types, list) and (var_type.lower() in factor_types)):
                i[1] = 'factor'
            else:
                assert var_type in ['numeric', 'factor'], \
                    'the type of %s is neither numeric nor factor' % i[0]

    return type_lst


# when importing data into dataframe, it will check if col names and variable names in dict are consist
def check_varname():
    """
    check if column names and variable names in dictionary are consistent
    :return: 
    """
    pass


def check_data_vol(data_df, id_name, target_name, row_low_limit=50, col_low_limit=1):
    """
    check if data set is big enough
    :param: data_df: a data frame
    :param: row_low_limit: lower limit of number of rows
    :param: col_low_limit: lower limit of number of cols
    :return: if data vol qualified
    """
    df_shape = data_df.shape
    assert df_shape[0] >= row_low_limit, 'there should be enough samples !'
    id_row = 1 if id_name is not None else 0
    target_row = 1 if target_name is not None else 0
    assert (df_shape[1] - (id_row + target_row)) >= col_low_limit, 'there should be enough variables !'

    return True



