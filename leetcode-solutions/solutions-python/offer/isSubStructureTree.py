# -*- coding: utf-8 -*-
# @Time : 2020/7/10 8:52
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : isSubStructureTree
# @Description: 树的子结构

"""
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
B是A的子结构， 即 A中有出现和B相同的结构和节点值。
"""
"""
解法：
1、前序遍历树A中的每一个节点Na：函数isSubStructure
2、判断树A中以Na为根节点的子树是否包含树B：函数recur
recur()
终止条件：节点B为空，B已经匹配完成，返回true
         节点A为空，越过A叶子节点，返回false
         节点A和B的值不同，匹配失败，返回false
返回值：判断A和B的左子节点是否相同，即recur(A.left,B.left)
       判断A和B的右子节点是否相同，即recur(A.right,B.right)
isSubStructure()
特例：当树A为空或树B为空时，直接返回 false
返回值： 若树B是树A的子结构，则必满足以下三种情况之一，因此用或 || 连接；
1、以节点A为根节点的子树包含树B，对应recur(A, B)；
2、树B是树A左子树的子结构，对应isSubStructure(A.left, B)；
3、树B是树A右子树的子结构，对应isSubStructure(A.right, B)；
"""


class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def recur(A, B):
            if not B: return True
            if not A or A.val != B.val: return False
            return recur(A.left, B.left) and recur(A.right, B.right)

        return bool(A and B) and (recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B))
