mport numpy as np

class node(object):
	def __init__(self, val):
		self.val = val
		self.left = None
		self.right = None
	
	def set_left(self, left):
		self.left = node(left)
	
	def set_right(self, right):
		self.right = node(right)
		
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
		allocated_space = -1e14*np.ones((height, allocated_len))
		
		def print_tree_helper(n, l, r, height, allocated_space):
			if n is None:
				return

			pos = int((r+l)/2.)
			allocated_space[height, pos] = n.get_val()
			print_tree_helper(n.left, l, pos-1, height+1, allocated_space)
			print_tree_helper(n.right, pos+1, r, height+1, allocated_space)
			return
			
		out_val = print_tree_helper(self, 0, allocated_len-1, 0, allocated_space)
		for i in allocated_space:
			for j in range(len(i)):
				if i[j] == -1e14:
					i[j] = None
				i[j] = str(i[j])
			print(i)
		
x = node(3)
x.set_left(5)
x.set_right(7)
x.left.set_left(2)
x.left.set_right(4)
x.right.set_left(1)
x.right.set_right(12)
x.print_tree()
