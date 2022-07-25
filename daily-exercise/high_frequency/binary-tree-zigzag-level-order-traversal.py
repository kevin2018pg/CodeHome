# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 21:31
# @Author  : kevin
# @Version : python 3.7
# @Desc    : binary-tree-zigzag-level-order-traversal
# 区分奇偶层序遍历树
# 给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）

# Definition for a binary tree node.
import copy


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def zigzagLevelOrder(self, root):
        if not root:
            return []

        level_node = [root]
        res_val = []
        level_num = 0
        while level_node:
            level_val = []
            new_level_node = []
            for node in level_node:
                if not node:
                    continue
                level_val.append(node.val)
                if node.left:
                    new_level_node.append(node.left)
                if node.right:
                    new_level_node.append(node.right)
            level_node = new_level_node
            if level_num % 2 == 1:
                new_level_val = self.define_reverse(level_val)
                res_val.append(new_level_val)
            else:
                res_val.append(level_val)
            level_num += 1
        return res_val

    def define_reverse(self, level):
        new_level = copy.deepcopy(level)
        length = len(new_level)
        for i in range(length // 2 - 1, -1, -1):
            j = length - 1 - i
            new_level[i], new_level[j] = new_level[j], new_level[i]
        return new_level
