# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 20:28
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 二叉树的最近公共祖先

# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
# 最近公共祖先的定义为：对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。

# 题解：若root是p，q的最近公共祖先，必然满足如下情况之一
# 1、p和q在 root的子树中，且分列root的异侧（即分别在左、右子树中）；
# 2、p=root，且q在root的左或右子树中；
# 3、q=root，且p在root的左或右子树中；
# 采用递归二叉树先序遍历

# 终止条件：
# 当越过叶节点，则直接返回root；
# 当root等于p,q，则直接返回root；
# 递推工作：
# 开启递归左子节点，返回值记为left；
# 开启递归右子节点，返回值记为right；
# 返回值：根据left和right ，可展开为四种情况；
# 当left和right同时为空：说明root的左/右子树中都不包含p,q，返回null；
# 当left和right同时不为空：说明p,q分列在root的异侧（分别在左/右子树），因此root为最近公共祖先，返回root；
# 当left为空，right不为空：p,q都不在root 的左子树中，直接返回right。具体可分为两种情况：
#   p,q 其中一个在root的右子树中，此时right指向p（假设为p）；
#   p,q 两节点都在root的右子树中，此时的right指向最近公共祖先节点 ；
# 当left不为空，right为空：与情况 3.同理；


class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right:
            return  # 1.
        if not left:
            return right  # 3.
        if not right:
            return left  # 4.
        return root  # 2. if left and right:
