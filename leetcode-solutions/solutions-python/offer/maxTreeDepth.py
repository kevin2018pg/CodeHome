# -*- coding: utf-8 -*-
# @Time : 2020/7/14 8:50
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : maxTreeDepth
# @Description: 二叉树最大深度


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    # DFS：左右子树最大深度+1
    def maxDepth(self, root: TreeNode) -> int:
        # 终止条件：root为空，越过叶子节点
        if not root: return 0
        # 递推：分别求左右子树的最大深度
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
