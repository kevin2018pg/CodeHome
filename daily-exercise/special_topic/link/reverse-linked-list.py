# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 14:06
# @Author  : kevin
# @Version : python 3.7
# @Desc    : reverse-linked-list
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head):
        if not head:
            return None

        pre, cur = None, head
        while cur:
            nex = cur.next
            cur.next = pre
            pre, cur = cur, nex
        return pre
