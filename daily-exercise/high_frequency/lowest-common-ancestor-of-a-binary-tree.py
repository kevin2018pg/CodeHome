# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 20:28
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 二叉树的最近公共祖先

# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
# 最近公共祖先的定义为：对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right: return  # 1.
        if not left: return right  # 3.
        if not right: return left  # 4.
        return root  # 2. if left and right:
