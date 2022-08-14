# -*- coding: utf-8 -*-
# @Time    : 2022/8/14 15:13
# @Author  : kevin
# @Version : python 3.7
# @Desc    : reverse-linked-list-ii

# 翻转链表2
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 方法：拿到left-right区间链表断开连接，局部翻转（调用翻转方法），拿到左前，右后节点，重新建立新连接即可
class Solution:
    def reverseBetween(self, head, left, right):
        if not head:
            return None
        # 虚拟头
        preHead = ListNode(next=head)
        leftBefore = preHead
        # 左前节点
        for _ in range(left - 1):
            leftBefore = leftBefore.next
        # 创建左右区间链表
        leftNode, rightNode = leftBefore.next, leftBefore
        for _ in range(right - left + 1):
            rightNode = rightNode.next
        # 断开连接
        rightAfter = rightNode.next
        leftBefore.next = None
        rightNode.next = None
        # 翻转链表
        self.reverse_link(leftNode)
        # 简历新连接
        leftBefore.next = rightNode
        leftNode.next = rightAfter

        return preHead.next

    def reverse_link(self, head):
        pre, cur = None, head
        while cur:
            nex = cur.next
            cur.next = pre
            pre, cur = cur, nex
