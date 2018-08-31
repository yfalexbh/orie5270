from pyspark import SparkContext
from operator import add
import os


def mul(a, B):
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
    num_col = B.count()
    mat = A.zipWithIndex().cartesian(B.zipWithIndex())
    matMultiply = mat.map(lambda (val1, val2): (val1[1], mul(val1[0][val2[1]], val2[0])))
    matReduce = matMultiply.reduceByKey(lambda a, b: list(map(add, a, b)))
    return matReduce.map(lambda (key, val): val)


if __name__ == '__main__':
    sc = SparkContext()
    out_mat = matrix_multiply('A.txt', 'B.txt', sc).collect()
    print(out_mat)
    file = open('result.txt', 'w')
    for i in out_mat:
        file.write(','.join([str(x) for x in i]) + '\n')
    file.close()