import pyspark as ps
import pandas as pd


def _text_file_helper(name_txt_file, sc):
    """
    process the input text file
    param name_txt_file: the name of the text file in string
    param sc: pre-defined SparkContext
    return: the processed data in dictionary
    """
    # read in the text file as a DataFrame object
    text = sc.textFile(name_txt_file)
    out_val = text.map(lambda x: x.split(' '))
    return out_val


if __name__ == '__main__':
    # create a SparkContext
    sc = SparkContext('KMeans')
    data = _text_file_helper('data.txt')
    centroids = _text_file_helper('c1.txt')
    print(data)

