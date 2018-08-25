from queue import PriorityQueue


def find_shortest_path(name_txt_file, source, destination):
    """
    Find the shortest path from source to destination using Dijkstra's algorithm
    param name_txt_file: name of the text file in string
    param source: the label of source vertex
    param destination: the label of destination vertex
    return: a tuple, the shortest path in a list and the length of that path
    """

def _read_txt_helper(name_txt_file):
    """
    read in the specific text file for a graph
    return: a dictionary maintaining the graph
    """
    d = {}
    file = open(name_txt_file, 'r')
    for line_index, line in enumerate(file.readlines()):
        if not line_index % 2:
            d[line] = []
        else:
            temp_list = line.split(',')
            # if only 2 elements, then there is only one tuple
            if len(temp_list) == 2:
                d[line] = [tuple(temp_list[0][1:], float(temp_list[1][:-1]))]
            # multiple tuples
            else:
                if len(temp_list) % 2:
                    raise Exception('wrong input, check text file' + name_txt_file)
                d[line] = []
                for i in xrange(0, len(temp_list), 2):
                    d[line].append(tuple(temp_list[i][1:], float(temp_list[i+1][:-1])))

    return d

