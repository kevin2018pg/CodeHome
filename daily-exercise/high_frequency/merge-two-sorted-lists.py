# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 9:29
# @Author  : kevin
# @Version : python 3.7
# @Desc    : merge-two-sorted-lists

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, list1, list2):
        preHead = ListNode(-1)
        tmpNode = preHead
        while list1 and list2:
            if list1.val <= list2.val:
                tmpNode.next = list1
                list1 = list1.next
            elif list1.val > list2.val:
                tmpNode.next = list2
                list2 = list2.next
            tmpNode = tmpNode.next
        if not list1:
            tmpNode.next = list2
        else:
            tmpNode.next = list1
        return preHead.next
