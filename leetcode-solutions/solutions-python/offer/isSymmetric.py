# -*- coding: utf-8 -*-
# @Time : 2020/7/13 20:16
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : isSymmetric
# @Description: 判断对称的二叉树

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def recur(L, R):
            # 终止条件
            # L和R同时为空，代表同时越过叶节点，此树从顶至底节点都对称返回true
            if not L and not R: return True
            # L和R中只有一个越过叶节点，或者节点L值不等于R值，此树不对称返回false
            if not L or not R or L.val != R.val: return False
            # 递推
            # 判断L左节点和R右节点，L右节点和R左节点是否对称，两节点都对称时，才是对称树，用and连接
            return recur(L.left, R.right) and recur(L.right, R.left)

        # 若根节点为空，返回true
        return recur(root.left, root.right) if root else True
