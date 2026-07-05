import tree_visualiser, random
class TreeNode:
    def __init__(self, val: int, left=None, right=None, depth=None):
        self.val = val
        self.left = left
        self.right = right
        self.depth = depth
        self.balance = 0
    def __str__(self):
        return self.val

class BinaryTree:
    def __init__(self, nums):
        self.root = TreeNode(nums[0])
        for i in nums[1:]:
            self.insert(self.root, i)
        self.update_balance_factor(self.root)
    
    def __getattr__(self, name):
        return getattr(self.root, name)
        


    def is_leaf(self, node):
        return node != None and node.left == None and node.right == None
        
    def remove(self, item: TreeNode):
        def find(head: TreeNode, item):
            if head:
                if head.val > item:
                    find(head.left, item)
                if head.val < item:
                    find(head.right, item)
                else:
                    pass
    
    def insert(self, root: TreeNode, val, depth=0):
        if self.is_leaf(root) and isinstance(root, TreeNode):
            if root.val > val:
                root.left = TreeNode(val, None, None, depth + 1)
                self.root = self.balance_tree()
            if root.val < val:
                root.right = TreeNode(val, depth=depth + 1)
                self.root = self.balance_tree()
            if root.val == val:
                raise ValueError(f"{val} already exists in the BST")
            depth = 0
        else:
            if root.val < val:
                if root.right:
                    self.insert(root.right, val, depth + 1)
                else:
                    root.right = TreeNode(val=val, depth = depth + 1)
                    self.root = self.balance_tree()
            elif root.val > val:
                if root.left:
                    self.insert(root.left, val, depth + 1)
                else:
                    root.left = TreeNode(val=val, depth=depth + 1)
                    self.root = self.balance_tree()
            else:
                raise ValueError(f"{val} already exists in the BST")
        
    def height(self, node):
        if node is None:
            return 0

        return 1 + max(self.height(node.left), self.height(node.right))


    def update_balance_factor(self, node:TreeNode):
        node.balance = self.height(node.left) - self.height(node.right)
        if node.left:
            self.update_balance_factor(node.left)
        if node.right:
            self.update_balance_factor(node.right)


    def inorder(self):
        def recurse(root, out = []):
            if root:
                recurse(root.left, out)
                out.insert(root.val)
                recurse(root.right, out)
        out = []
        recurse(self.root, out)
        return out
    
    def preorder(self):
        def recurse(root, out = []):
            if root:
                out.insert(root.val)
                recurse(root.left, out)
                recurse(root.right, out)
        out = []
        recurse(self.root, out)
        return out
    
    def postorder(self):
        def recurse(root, out = []):
            if root:
                recurse(root.left, out)
                recurse(root.right, out)
                out.insert(root.val)
        out = []
        recurse(self.root, out)
        return out

    def balance_tree(self):
        def rotate_right(node: TreeNode):
            x = node.left
            t2 = x.right
            x.right = node
            node.left = t2
            x.depth = height(x)
            node.depth = height(node)
            self.update_balance_factor(node)
            self.update_balance_factor(x)
            return x
        
        def rotate_left(node: TreeNode):
            y = node.right
            t2 = y.left
            y.left = node
            node.right = t2
            y.depth = height(y)
            node.depth = height(node)
            self.update_balance_factor(node)
            self.update_balance_factor(y)
            return y

        def balance(root: TreeNode): #balance subtree
            self.update_balance_factor(root)
            if root.balance == 0:
                return root
            elif root.balance > 1:
                if root.left.balance >= 0:
                    #left-left
                    root = rotate_right(root)
                elif root.left.balance < 0:
                    #left-right
                    root.left = rotate_left(root.left)
                    root = rotate_right(root)
                
            elif root.balance < -1:
                if root.right.balance <= 0:
                    #right-right
                    root = rotate_left(root)
                elif root.right.balance > 0:
                    #right-left
                    root.right = rotate_right(root.right)
                    root = rotate_left(root)
            return root
        #postorder
        def height(node):
            if node is None:
                return 0

            return 1 + max(height(node.left), height(node.right))
        def recurse(node: TreeNode):
            if node:
                node.left = recurse(node.left)
                node.right = recurse(node.right)

                node.depth = height(node)

                return balance(node)
        self.root = recurse(self.root)
        return self.root

nums = list(range(0, 100))
random.shuffle(nums)
a = BinaryTree(nums)
# print(a.root.val, [a.root.right.val, a.root.right.depth], [a.root.left.val, a.root.left.depth], a.root.balance)
print(vars(a.root), a.postorder())
a.insert(a.root, 101)
# tree_visualiser.visualize_bst(a.root)
print(vars(a.root), a.postorder(), a.val)
print(a.root.val)

        