from pyspark.sql import SparkSession


def hive_to_hdfs(ss, hive, hdfs):
    data = ss.sql("select * from %s" % hive)
    data.write.csv(hdfs, header=True, mode='overwrite')


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .enableHiveSupport() \
        .getOrCreate()
    hive_to_hdfs(spark, "default.overdue", "hdfs://ip-172-31-22-94.cn-north-1.compute.internal:8020/overdue.csv")
