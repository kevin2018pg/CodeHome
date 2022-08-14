# -*- coding: utf-8 -*-
# @Time    : 2022/8/14 18:11
# @Author  : kevin
# @Version : python 3.7
# @Desc    : remove-duplicates-from-sorted-list-ii
# 删除重复节点进阶，不保留重复元素

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def deleteDuplicates(self, head):
        if not head:
            return None
        preHead = ListNode(next=head)
        cur = preHead
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                x = cur.next.val
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return preHead.next
