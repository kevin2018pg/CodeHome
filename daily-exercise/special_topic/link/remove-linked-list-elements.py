# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 12:44
# @Author  : kevin
# @Version : python 3.7
# @Desc    : remove-linked-list-elements
# 删除链表指定值元素
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeElements(self, head, val: int):
        prehead = ListNode(next=head)  # 虚拟头节点，返回值需要虚拟头节点的next节点
        cur = prehead  # 移动节点
        while cur.next != None:
            if cur.next.val == val:  # 下个节点值是否满足
                cur.next = cur.next.next  # 删除节点再判断
            else:
                cur = cur.next  # 满足，移动节点
        return prehead.next
