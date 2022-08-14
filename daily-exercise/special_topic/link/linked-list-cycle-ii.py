# -*- coding: utf-8 -*-
# @Time    : 2022/8/14 14:49
# @Author  : kevin
# @Version : python 3.7
# @Desc    : linked-list-cycle-ii

# 环形链表2
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


#  哈希表
class Solution:
    def detectCycle(self, head):
        if not head:
            return None
        hash_node = {}
        while head:
            if head in hash_node:
                return head
            hash_node[head] = True
            head = head.next
        return None
