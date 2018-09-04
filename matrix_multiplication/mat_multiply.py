from pyspark import SparkContext
from operator import add
from operator import mul
import os
from sys import argv


def multi(a, B):
    return [a*x for x in B]


def matrix_multiply(name_txt_file1, name_txt_file2, sc):
    """
    matrix multiplication in Spark
    param name_txt_file1: name of the first text file (matrix) in string
    param name_txt_file2: name of the second text file (matrix) in string
    param sc: a SparkContext
    return: the name of the output text file in the current directory
    """
    A = sc.textFile(name_txt_file1).map(lambda x: [float(y) for y in x.split(',')]).cache()
    B = sc.textFile(name_txt_file2).map(lambda x: [float(y) for y in x.split(',')]).cache()
    if len(B.collect()) == 1:
        # matrix-vector
        B = sc.textFile(name_txt_file2).map(lambda x: [float(y) for y in x.split(',')]).collect()
        matMultiply = A.zipWithIndex().map(lambda (x, key): (key, sum(list(map(mul, x, B[0])))))
        return matMultiply.map(lambda (key, val): val)

    else:
        mat = A.zipWithIndex().cartesian(B.zipWithIndex())
        matMultiply = mat.map(lambda (val1, val2): (val1[1], multi(val1[0][val2[1]], val2[0])))
        matReduce = matMultiply.reduceByKey(lambda a, b: list(map(add, a, b)))
        return matReduce.map(lambda (key, val): [val])


if __name__ == '__main__':
    sc = SparkContext()
    out_mat = matrix_multiply(argv[1], argv[2], sc).collect()
    print(out_mat)
