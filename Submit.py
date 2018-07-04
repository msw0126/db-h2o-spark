# -*- coding:utf-8 -*-
"""
提交任务
"""

import os, subprocess, sys
reload(sys)
sys.setdefaultencoding('utf8')


SPARK_PATH = "F:\\tools\\Spark\\spark-2.1.0-bin-hadoop2.7\\bin\\spark-submit"
HADOOP_CONFIG = "F:\\tools\\Spark\\hadoop_config"
HADOOP_USER_NAME = "hdfs"
SPARK_CLASSPATH = "F:\\tools\\Spark\\spark-2.1.0-bin-hadoop2.7\\jars"


os.environ.setdefault("HADOOP_CONF_DIR", HADOOP_CONFIG)
os.environ.setdefault("HADOOP_USER_NAME", HADOOP_USER_NAME)
os.environ.setdefault("YARN_CONF_DIR", HADOOP_CONFIG)
os.environ.setdefault("SPARK_CLASSPATH", SPARK_CLASSPATH)

command = [
        SPARK_PATH,
        "--master", "yarn",
        "--deploy-mode", "cluster",
        "--name", "Atom-Test",
        "--files", "F:\\learn\\db-h2o-spark\\tdir\\learn_config.json,"
                   "F:\\learn\\db-h2o-spark\\tdir\\hive_reader_dict.csv,"
                   "F:\\learn\\db-h2o-spark\\tdir\\act_config.json,"
                   "F:\\learn\\db-h2o-spark\\tdir\\test_config.json",
        "--py-files", "Atom.zip,backports.inspect-0.0.3.tar.gz,certifi.zip,chardet.zip,colorama.zip,future.zip,h2o_pysparkling_2.1-2.1.17.zip,idna.zip,pkg_resources.py,prettytable-0.7.2.zip,pytz.zip,requests.zip,tabulate.zip,traceback2-1.4.0.zip,urllib3-1.22.zip",
        "--driver-memory", "1G",
        "--num-executors", "1",
        "--executor-memory", "1G",
        "F:\\learn\\db-h2o-spark\\Atom.py", "Act"
    ]
print( " ".join( command ) )
print( os.path.dirname( os.path.realpath( sys.argv[0] ) ) )
try:
    p = subprocess.Popen(" ".join( command ),
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd=os.path.dirname(os.path.realpath(sys.argv[0])))
    application_id = None
    tracking_url = None
    while p.poll() is None:
        # print p.poll()
        line = p.stderr.readline()#.decode('utf-8', 'ignore').strip()
        print line
        if len(line) > 0 and (application_id is None or tracking_url is None):
            assert isinstance(line, str)
            if line.startswith("tracking URL:"):
                tracking_url = line.replace("tracking URL:", "").strip()
                print(tracking_url)
            elif "Submitted application" in line:
                application_id = line.split("Submitted application")[1].strip()
                print(application_id)
except Exception as e:
    # print( str( e ).decode( 'cp936' ).encode( 'utf-8' ) )
    print( e )
