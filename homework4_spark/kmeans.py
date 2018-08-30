from pyspark import SparkContext
import pandas as pd
import numpy as np
import numpy.linalg as npla
from operator import add
from operator import neg


def spark_KMeans(points, centroids, max_iter=100):
    """
    kmeans in spark
    param points: data points
    param centroids: initial centroids for K-means
    """
    for i in range(max_iter):
        interm = points.map(lambda x: (np.argmin([npla.norm(list(map(add, x, list(map(neg, j))))) for j in centroids]), (x, 1)))
        centroids = interm.reduceByKey(lambda a, b: (list(map(add, a[0], b[0])), a[1] + b[1])).map(lambda (key, x): [y / x[1] for y in x[0]]).collect()
    return centroids


def _text_file_helper(name_txt_file, sc):
    """
    process the input text file
    param name_txt_file: the name of the text file in string
    param sc: pre-defined SparkContext
    return: the processed data in dictionary
    """
    # read in the text file as a DataFrame object
    text = sc.textFile(name_txt_file)
    out_val = text.map(lambda x: [float(y) for y in x.split(' ')])
    return out_val


if __name__ == '__main__':
    # create a SparkContext
    sc = SparkContext('local')
    data = _text_file_helper('data.txt', sc).cache()
    centroids = _text_file_helper('c1.txt', sc).collect()
    out_centroids = spark_KMeans(data, centroids)
    file = open('result.txt', 'w')
    
    for i in out_centroids:
        file.write(' '.join([str(round(j, 3)) for j in i]) + '\n')
    file.close()
