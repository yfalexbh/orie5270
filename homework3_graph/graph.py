import heapq


def negative_cycle(name_txt_file):
    """
    Find a negative cycle in the graph using Bellman-Ford algorithm
    param name_txt_file: the name of the text file in string
    return: negative cycle in a list
    """


def find_shortest_path(name_txt_file, source, destination):
    """
    Find the shortest path from source to destination using Dijkstra's algorithm
    param name_txt_file: name of the text file in string
    param source: the label of source vertex
    param destination: the label of destination vertex
    return: a tuple, the shortest path in a list and the length of that path
    """
    # type conversion
    source = str(source)
    destination = str(destination)
    d = _read_txt_helper(name_txt_file)
    visited = set()
    # first element is the current shortest path, the second is the current node, and the last is the path
    p_queue = [(0, source, [])]
    dist_to_node = {}
    while p_queue:
        ele = heapq.heappop(p_queue)
        print(ele)
        if ele[1] not in visited:
            visited.add(ele[1])
            path = ele[2]

            if ele[1] == destination:
                return path + [ele[1]], ele[0]

            for i in d[ele[1]]:
                if i[0] not in visited:
                    dist = float(i[1]) + ele[0]
                    if i[0] not in dist_to_node or dist < dist_to_node[i[0]]:
                        dist_to_node[i[0]] = dist
                        heapq.heappush(p_queue, (dist, i[0], path + [ele[1]]))


def _read_txt_helper(name_txt_file):
    """
    read in the specific text file for a graph
    return: a dictionary maintaining the graph
    """
    d = {}
    file = open(name_txt_file, 'r')
    f = file.readlines()
    for l in xrange(0, len(f), 2):
        key = f[l].rstrip()
        d[f[l].rstrip()] = []
        temp_list = f[l+1].rstrip().split(',')
        # no element
        if len(temp_list) == 1:
            pass
        # if only 2 elements, then there is only one tuple
        elif len(temp_list) == 2:
            d[key] = [(temp_list[0][1:], float(temp_list[1][:-1]))]
        # multiple tuples
        else:
            if len(temp_list) % 2:
                raise Exception('wrong input, check text file' + name_txt_file)
            d[key] = []
            for i in xrange(0, len(temp_list), 2):
                d[key].append((temp_list[i][1:], float(temp_list[i+1][:-1])))

    return d

if __name__ == '__main__':
    print(find_shortest_path('graph.txt', 1, 5))
