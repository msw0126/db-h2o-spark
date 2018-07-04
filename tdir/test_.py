# -*- coding:utf-8 -*-
import json

import h2o

from BaseModules.FileIO import byteify


def parse_json2dict_upload(file_name):
    """
    parse a json file uploaded into a dictionary
    :param: file_name: upload json file name
    :return: a parsed dictionary
    """
    with open(file_name) as dict_file:
        dict_content = dict_file.read()
        print dict_content
    print "--------------------"
    print json.loads(dict_content)
    print "-----------------------"
    print byteify(json.loads(dict_content, object_hook=byteify), ignore_dicts=True)
    # return byteify(json.loads(dict_content, object_hook=byteify), ignore_dicts=True)


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



# print parse_json2dict_upload("./learn_config.json")
# parse_json2dict_upload("./learn_config.json")

if __name__ == '__main__':
    dict_path = "hdfs://node1:8020/taoshu/engine/work_dir/103/RobotXSpark7/dict.csv"
    parse_csv2list_hdfs(dict_path)