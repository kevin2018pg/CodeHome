# -*- coding: utf-8 -*-
# @Time    : 2022/8/14 17:32
# @Author  : kevin
# @Version : python 3.7
# @Desc    : reorder-list
# 重排链表
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 用线性表存储起来，头尾访问即可
class Solution:
    def reorderList(self, head):
        """
        Do not return anything, modify head in-place instead.
        """
        stack_node = list()
        while head:
            stack_node.append(head)
            head = head.next
        i, j = 0, len(stack_node) - 1
        while i < j:
            stack_node[i].next = stack_node[j]
            i += 1
            if i == j:
                break
            stack_node[j].next = stack_node[i]
            j -= 1
        stack_node[i].next = None
