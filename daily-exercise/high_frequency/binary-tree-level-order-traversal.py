# -*- coding: utf-8 -*-
# @Time    : 2022/7/22 17:53
# @Author  : kevin
# @Version : python 3.7
# @Desc    : binary-tree-level-order-traversal

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 基于BFS方法：一层一层从左向右选择节点和值加入，注意从左到右的顺序
class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        res_val = list()
        level_node = [root]  # 初始化加入root节点
        while level_node:  # 遍历条件，当前层node一直存在
            level_val = list()  # 当前层从左到右的值
            new_leve_node = list()  # 用新的node集合存储，原集合需要删除上一层node节点，比较麻烦
            for node in level_node:  # 遍历node，计入node的val，新node集合加入左右node节点
                if not node:
                    continue
                level_val.append(node.val)
                if node.left:
                    new_leve_node.append(node.left)
                if node.right:
                    new_leve_node.append(node.right)
            res_val.append(level_val)  # 层级值集合加入总集合
            level_node = new_leve_node  # 新层级节点赋给层节点，保持遍历层条件
        return res_val
