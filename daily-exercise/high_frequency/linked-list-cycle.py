# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 16:06
# @Author  : kevin
# @Version : python 3.7
# @Desc    : linked-list-cycle

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head):
        hash_link = set()
        while head:
            if head in hash_link:
                return True
            hash_link.add(head)
            head = head.next
        return False
