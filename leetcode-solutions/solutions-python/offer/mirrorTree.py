# -*- coding: utf-8 -*-
# @Time : 2020/7/13 8:52
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : mirrorTree
# @Description: 二叉树镜像

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 递归
    def mirrorTree_s1(self, root: TreeNode) -> TreeNode:
        # 终止条件
        if not root: return
        # 交换左右子节点
        root.left, root.right = root.right, root.left
        # 递归
        self.mirrorTree_s1(root.left)
        self.mirrorTree_s1(root.right)
        # 返回
        return root

    # 递归
    def mirrorTree_s2(self, root: TreeNode) -> TreeNode:
        # 当节点root为空时（即越过叶节点），则返回null
        if not root: return
        # 递归遍历（dfs）二叉树，交换左右子节点
        root.left, root.right = self.mirrorTree_s2(root.right), self.mirrorTree_s2(root.left)
        # 返回当前节点root
        return root

    # 辅助栈方法
    def mirrorTree_s3(self, root: TreeNode) -> TreeNode:
        if not root: return
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left: stack.append(node.left)
            if node.right: stack.append(node.right)
            node.left, node.right = node.right, node.left
        return root
