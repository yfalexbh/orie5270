# a tree structure
class Tree(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def set_left(self, left):
        self.left = Tree(left)

    def set_right(self, right):
        self.right = Tree(right)

    def get_height(self):
        if self is None:
            return 0
        elif self.left is None and self.right is None:
            return 1
        elif self.left is None:
            return 1 + self.right.get_height()
        elif self.right is None:
            return 1 + self.left.get_height()

        return 1 + max(self.left.get_height(), self.right.get_height())

    def get_val(self):
        return self.val

    def print_tree(self):

        height = self.get_height()
        allocated_len = 2**height - 1
        allocated_space = [['|'] * allocated_len]
        for i in range(height - 1):
            allocated_space.append([copy for copy in allocated_space[0]])

        allocated_space = Tree.print_tree_helper(self, 0, allocated_len-1, 0, allocated_space)
        for i in allocated_space:
            for j in (i):
                print(j),
            print

        return allocated_space

    @staticmethod
    def print_tree_helper(n, l, r, height, allocated_space):
        if n is None:
            return allocated_space

        pos = int((r+l)/2.)
        allocated_space[height][pos] = str(n.get_val())
        allocated_space = Tree.print_tree_helper(n.left, l, pos-1, height+1, allocated_space)
        allocated_space = Tree.print_tree_helper(n.right, pos+1, r, height+1, allocated_space)

        return allocated_space
