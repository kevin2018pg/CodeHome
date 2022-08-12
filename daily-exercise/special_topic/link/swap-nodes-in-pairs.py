# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 15:07
# @Author  : kevin
# @Version : python 3.7
# @Desc    : swap-nodes-in-pairs
# 两两交换链表节点

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head):
        # 翻转两个局部链表节点需要用到三个节点，因为需要更新前后的指向，所以创建一个虚拟头，更新虚拟头
        prehead = ListNode(next=head)
        before = prehead  # 三节点之前节点
        while before.next and before.next.next:
            middle = before.next  # 三节点之中节点
            after = before.next.next  # 三节点之后节点

            # 开始翻转（先翻转中后再改变前的指向）
            middle.next = after.next
            after.next = middle
            before.next = after

            before = before.next.next  # 更新前节点（虚拟头）
        return prehead.next
