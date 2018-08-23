import unittest
from tree.Tree import Tree


class testTree(unittest.TestCase):
    def test_get_height_1(self):
        root = Tree(2)
        assert root.get_height() == 1

    def test_get_height_2(self):
        root = Tree(2)
        root.set_left(3)
        assert root.get_height() == 2

    def test_get_height_3(self):
        root = Tree(2)
        root.set_right(3)
        assert root.get_height() == 2

    def test_one_node_tree(self):
        assert Tree(2).print_tree() == [['2']]

    def test_balanced_tree(self):
        root = Tree(1)
        root.set_left(4)
        root.set_right(5)
        root.left.set_left(3)
        root.left.set_right(5)
        root.right.set_left(7)
        root.right.set_right(9)
        answer = [['|', '|', '|', '1', '|', '|', '|']]
        answer.append(['|', '4', '|', '|', '|', '5', '|'])
        answer.append(['3', '|', '5', '|', '7', '|', '9'])

        assert root.print_tree() == answer

    def test_imbalanced_tree(self):
        root = Tree(6)
        root.set_left(5)
        root.left.set_left(2)
        root.left.set_right(3)
        answer = [['|', '|', '|', '6', '|', '|', '|']]
        answer.append(['|', '5', '|', '|', '|', '|', '|'])
        answer.append(['2', '|', '3', '|', '|', '|', '|'])

        assert root.print_tree() == answer
