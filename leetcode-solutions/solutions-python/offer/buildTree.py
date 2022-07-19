# -*- coding: utf-8 -*-
# @Time : 2020/7/6 8:26
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : buildTree
# @Description: 重建二叉树

"""
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树
假设输入的前序遍历和中序遍历的结果中都不含重复的数字，即每个节点唯一
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回：
    3
   / \
  9  20
    /  \
   15   7
"""


# 利用原理,先序遍历的第一个节点就是根。在中序遍历中通过根 区分哪些是左子树的，哪些是右子树的

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 分别标记中序遍历和保留前序遍历
        self.dic, self.po = {}, preorder
        # 为了提升搜索效率，使用哈希表 dic 预存储中序遍历的值与索引的映射关系，每次搜索的时间复杂度为 O(1)
        for i in range(len(inorder)):
            self.dic[inorder[i]] = i
        return self.recur(0, 0, len(inorder) - 1)

    # 递推参数：前序遍历中根节点的索引pre_root、中序遍历左边界in_left、中序遍历右边界in_right
    def recur(self, pre_root, in_left, in_right):
        if in_left > in_right: return  # 终止条件：中序遍历为空，说明已经越过叶子节点（相等就是自己）
        root = TreeNode(self.po[pre_root])  # 建立当前子树的根节点
        i = self.dic[self.po[pre_root]]  # 搜索根节点在中序遍历中的索引，从而可对根节点、左子树、右子树完成划分
        # 左子树的根节点就是左子树的(前序遍历）第一个，就是 + 1（前序【根|左|右】）
        root.left = self.recur(pre_root + 1, in_left, i - 1)  # 开启左子树的下层递归，根节点索引为 pre_root + 1
        # 右子树的根节点就是右子树（前序遍历）的第一个,就是当前根节点+1 加上左子树长度
        root.right = self.recur(pre_root + 1 + (i - in_left), i + 1,
                                in_right)  # 开启右子树的下层递归，根节点索引为 i - in_left + pre_root + 1（即：左子树长度 + 根节点索引 + 1）
        return root  # 返回根节点，当前递归层级建立的根节点root为上一递归层级的根节点的左或右子节点
