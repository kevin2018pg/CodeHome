# -*- coding: utf-8 -*-
# @Time : 2020/7/9 18:58
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : mergeTwoList
# @Description: 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的


"""
根据题目描述,链表l1,l2是递增的，因此容易想到使用双指针l1和l2遍历两链表，
根据 l1.val和 l2.val的大小关系确定节点添加顺序，两节点指针交替前进，直至遍历完毕

初始状态合并列表中没有节点，因此循环第一轮时无法将节点添加到合并链表中，
所以引入伪头节点作为辅助节点，将各个几点加到其后
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 辅助节点迭代法
    def mergeTwoLists_s1(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 辅助的伪头结点dum,并用cur指向
        cur = dum = ListNode(0)
        # l1或l2为空时跳出
        while l1 and l2:
            # l1<l2，节点cur指向l1，并且l1向前走一步
            if l1.val < l2.val:
                cur.next, l1 = l1, l1.next
            # l1>=l2，节点cur指向l2，并且l2向前走一步
            else:
                cur.next, l2 = l2, l2.next
            # 移动cur,cur节点向前走一步
            cur = cur.next
        # 当跳出时,l1不为空就全部加到cur的后面,如果l1为空,就把l2加到cur后面
        cur.next = l1 if l1 else l2
        # 返回dum的后继节点
        return dum.next

    # 递归方法
    def mergeTwoLists_s2(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 前两个是递归终止条件
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists_s2(l1.next, l2)
            return l1
        l2.next = self.mergeTwoLists_s2(l1, l2.next)
        return l2
