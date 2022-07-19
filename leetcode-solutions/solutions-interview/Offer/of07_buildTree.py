'''
Author: your name
Date: 2020-11-17 08:30:17
LastEditTime: 2020-11-17 10:04:26
LastEditors: Please set LastEditors
Description: 剑指 Offer 07. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如给出：
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7


FilePath: \learningflow\Review\OfferSolution\of07_buildTree.py
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """"
        前序遍历性质： 节点按照 [ 根节点 | 左子树 | 右子树 ] 排序
        中序遍历性质： 节点按照 [ 左子树 | 根节点 | 右子树 ] 排序
        preorder-前序 inorder-中序
        """"

        def recur(root, left, right):
            if left > right: return  # 终止条件：中序遍历为空，说明已经越过叶子节点（相等就是自己）
            node = TreeNode(self.preorder[root])  # 建立当前子树根节点
            i = dic[self.preorder[root]]  # 搜索根节点在中序遍历中的索引，划分根节点、左子树、右子树
            node.left = recur(root + 1, left, i - 1)  # 开启左子树递归（根索引：root + 1）
            node.right = recur(i - left + root + 1, i + 1, right)  # 开启右子树递归（根索引：左子树长度 + 根索引 + 1）
            return node  # 回溯返回根节点，当前递归层级建立的根节点node为上一递归层级的根节点的左或右子节点

        # 提升搜索效率，dic 预存储中序遍历的值与索引的映射，O(1)
        dic, self.preorder = {}, preorder
        for i in range(len(inorder)):
            dic[inorder[i]] = i
        return recur(0, 0, len(inorder) - 1)
