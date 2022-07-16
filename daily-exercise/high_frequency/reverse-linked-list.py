# -*- coding: utf-8 -*-
# @Time    : 2022/3/12 17:38
# @Author  : kevin
# @File    : reverse-linked-list.py
# @Version : python 3.6
# @Desc    : 翻转链表

# 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
# 输入：head = [1,2,3,4,5]
# 输出：[5,4,3,2,1]

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# 双指针，先存再转再后移
class Solution:
    def reverseList(self, head):
        if not head:
            return head

        pre, cur = None, head
        while cur:
            nex = cur.next
            cur.next = pre
            pre, cur = cur, nex
        return pre


class Solution_digui:
    def reverseList(self, head):  # head-1-2-3-4-5
        if not head or not head.next:  # 5没有下一个节点,返回5
            return head
        new_head = self.reverseList(head.next)  # 逐轮出栈，第一轮head=5，第二轮head=4
        head.next.next = head  # 4.next=5,5.next=4
        head.next = None  # 4.next=None

        return new_head
