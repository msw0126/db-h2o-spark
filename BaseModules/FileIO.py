# -*- coding:utf-8 -*-

# The functions of this module include read and write files on hdfs or upload file, include:
#    1. write string into a (new)file on hdfs;
#    2. make a directory on hdfs;
#    3. parse a json file uploaded into a dictionary;
#    4. parse a json file on hdfs into a dictionary;
#    5. dump a dictionary to a json on hdfs;
#    6. parse a csv file uploaded into a list of lists;
#    7. parse a csv file on hdfs into a list of lists;
#    8. dump a list to a csv file on hdfs;
#    9. import data to a dataframe;
#    10. import dict to a list;
#

from pyspark import SparkFiles
import h2o
from DataAudit import *
import os
import json, csv
import ast
from pysparkling import *
from pyspark.sql import SparkSession


def path(sc, file_path):
    """
    create hadoop path object
    :param sc: sparkContext object
    :param file_path: file absolute path
    :return: org.apache.hadoop.fs.Path object
    """
    path_class = sc._gateway.jvm.org.apache.hadoop.fs.Path
    path_obj = path_class(file_path)
    return path_obj


def get_file_system(sc):
    """
    create FileSystem
    :param sc: SparkContext
    :return: FileSystem object
    """
    filesystem_class = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoop_configuration = sc._jsc.hadoopConfiguration()
    return filesystem_class.get(hadoop_configuration)


def write_to_hdfs(sc, file_path, content, overwrite=True):
    """
    wirte string into a (new)file on hdfs
    :param sc: SparkContext
    :param file_path: absolute path
    :param content: file content
    :param overwrite: whether overwrite 
    :return: nothing
    """
    try:
        filesystem = get_file_system(sc)
        out = filesystem.create(path(sc, file_path), overwrite)
        out.write(bytearray(content, "utf-8"))
        out.flush()
        out.close()
    except Exception as e:
        raise e


def make_dir(sc, dir_path):
    """
    make hdfs file directory
    :param sc: SparkContext
    :param dir_path: directory path
    :return: nothing
    """
    try:
        filesystem = get_file_system(sc)
        tmp = path(sc, dir_path)
        if not filesystem.exists(tmp):
            filesystem.mkdirs(tmp)
    except Exception as e:
        raise e


def byteify(data, ignore_dicts=False):
    """
    convert utf-8 string to string in dict from json
    :param data: dict data
    :param ignore_dicts:
    :return:
    """
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [byteify(item, ignore_dicts=True) for item in data]
    # if this is a dictionary, return dictionary of byteified keys and values(如果这是一个字典，返回字节化的键和值的字典)
    # but only if we haven't already byteified it(但前提是我们还没有对它进行过讨论。)
    if isinstance(data, dict) and not ignore_dicts:
        return {
            byteify(key, ignore_dicts=True): byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


def parse_json2dict_hdfs(sc, dir_name, file_name):
    """
    parse a json file on hdfs into a dictionary
    :param: sc: spark context
    :param: dir_name: directory name contain the json on hdfs
    :param: file_name: json file name
    :return: a parsed dictionary
    """
    file_path = os.path.join(dir_name, file_name)
    sc.addFile(file_path)
    with open(SparkFiles.get(file_name)) as json_file:
        content = json_file.read()

    return byteify(json.loads(content, object_hook=byteify), ignore_dicts=True)

    # return json.loads(content)
    # return ast.literal_eval(content)

def parse_json2dict_upload(file_name):
    """
    parse a json file uploaded into a dictionary
    :param: file_name: upload json file name
    :return: a parsed dictionary
    """
    with open(file_name) as dict_file:
        dict_content = dict_file.read()

    # 返回字节化的字典
    return byteify(json.loads(dict_content, object_hook=byteify), ignore_dicts=True)

    # return json.loads(dict_content)
    # return ast.literal_eval(dict_content)


def dump_dict2json_hdfs(sc, file_path, dict_content):
    """
    dump a dictionary to a json on hdfs
    :param: sc: spark contxt
    :param: file_path: path to save json file
    :return: nothing
    """
    write_to_hdfs(sc, file_path, json.dumps(dict_content, indent=1), overwrite=True)


def parse_csv2list_upload(file_name):
    """
    parse a csv file uploaded into a list of lists
    1. csv encoding: utf-8
    2. contents of list type: string
    :param: file_name: name of a csv file uploaded
    :return: a list of lists
    """
    with open(file_name) as f:
        records = csv.reader(f)
        csv_list = [[j.strip() for j in record] for record in records]
    return csv_list


def parse_csv2list_hdfs(file_path):
    """
    parse a csv file on hdfs into a list of lists
    1. csv encoding: utf-8
    2. contents of list type: string
    :param: file_path: path of a csv file on hdfs
    :return: a list of lists
    """
    file_df = h2o.import_file(path=file_path, header=1, sep=',')
    file_df = file_df[:, :2].ascharacter()
    csv_list = h2o.as_list(file_df, use_pandas=False, header=True)
    csv_list = [[j.strip() for j in i] for i in csv_list]
    return csv_list


def dump_list2csv_hdfs(sc, content_list, file_path):
    """
    convert a list of lists to a csv file on hdfs
    :param: sc: spark context
    :param: content_list: a list of lists
    :param: file_path: path to save csv file 
    :return: nothing
    """
    content_str = ''
    for i in range(len(content_list)):
        i_content = ','.join([str(col) for col in content_list[i]]) + '\n'
        content_str += i_content
    write_to_hdfs(sc, file_path, content_str, overwrite=True)


def import_data_dict(dict_path, numeric_types=None, factor_types=None):
    """
    import data type dict
    precondition:
        1. upload or on hdfs;
        2. csv format;
        3. two lines: variable and type;
        4. first line: 'variable' ,'type'
    function:
        1. check if the content qualified;
        2. parse it into a dict;
    :param: dict_path: dict file path on hdfs or filename if upload by submit command

    :return: type dict and col list
    """
    # import file into a list of lists;
    if dict_path is not None and len(str(dict_path)) > 0 and str(dict_path)[:5] == 'hdfs:':
        dict_lst = parse_csv2list_hdfs(str(dict_path))
    else:
        dict_lst = parse_csv2list_upload(str(dict_path))

    # -------
    tmp_lst = [i[:2] for i in dict_lst if str(i[1]) not in ['date', 'Date', 'DATE']]
    dict_lst = tmp_lst
    # -------
    print dict_lst

    # check and parse a list into dictionary;
    type_dict = dict()
    col_name_lst = list()
    if check_type_dict(dict_lst, numeric_types, factor_types):
        for i in dict_lst[1:]:
            if str(i[1]) != 'date':
                type_dict[str(i[0])] = str(i[1])
                col_name_lst.append(str(i[0]))

    return type_dict, col_name_lst


def import_data_as_frame(data_path, type_dict=None, sep=',', col_name_lst=None, header=True, na_lst=None):
    """
    import data from hdfs, and convert it into h2o data frame,
    1. create a dataframe dictionary with a dataframe, its schema and other information;
    2. make sure to create a dict if it doesn’t exist
    :param: data_path: data file path
    :param: sep: separator, default: ','
    :param: header: if header exists
    :param: type_dict: data type dict
    :param: na_lst: strings represent null
    :return: a data object with data frame, type dict and other info dict 
    """
    if type_dict is not None and header:
        data_df = h2o.import_file(path=data_path, header=1, sep=sep, col_types=type_dict,
                                  na_strings=na_lst)
        print '---- test debug import_data_as_frame 1---'
        print data_df.shape

        data_df = data_df[:, list(type_dict.keys())]

        print '---- test debug import_data_as_frame 2 ---'
        print data_df.shape

    elif type_dict is not None and not header:
        data_df = h2o.import_file(path=data_path, header=-1, sep=sep, col_names=col_name_lst, col_types=type_dict,
                                  na_strings=na_lst)
        data_df = data_df[:, list(type_dict.keys())]
    else:
        assert type_dict is None and header, \
            'headers needed if there is no type dict !'
        data_df = h2o.import_file(path=data_path, header=1, sep=sep, na_strings=na_lst)
        type_dict = data_df.types
        for key in type_dict.keys():
            if type_dict[key] in ['int', 'real']:
                type_dict[key] = 'numeric'
            else:
                type_dict[key] = 'factor'
                data_df[key] = data_df[key].asfactor()

    data_obj = {'df': data_df, 'schema': type_dict, 'info': dict()}

    return data_obj


def transfer_sparkdf_as_h2odf(data_df, type_dict=None):
    """
    """
    for var in type_dict.keys():
        if type_dict[var] == 'numeric':
            data_df[var] = data_df[var].asnumeric()
        elif type_dict[var] == 'factor':
            data_df[var] =data_df[var].ascharacter().asfactor()

    data_df = data_df[:, list(type_dict.keys())]
    data_obj = {'df': data_df, 'schema': type_dict, 'info': dict()}

    return data_obj


def hive_to_hdfs(spark, hive):
    """

    :param ss:
    :param hive:
    :param hdfs:
    :return:
    """
    data = spark.sql("select * from %s" % hive)
    data.show()
    hc = H2OContext.getOrCreate(spark)
    data_df = hc.as_h2o_frame(data)
    # data.show()
    # data.write.option("quote", "").csv(hdfs, header=True, mode='overwrite')
    return data_df


