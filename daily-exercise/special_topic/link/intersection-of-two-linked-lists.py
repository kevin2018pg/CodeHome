# -*- coding: utf-8 -*-
# @Time    : 2022/8/14 12:29
# @Author  : kevin
# @Version : python 3.7
# @Desc    : intersection-of-two-linked-lists

# 相交链表
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# 第一种方法等同于判断链表是否有环，把节点加入哈希表，判断是否存在即可
class SolutionHash:
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        hash_node = dict()
        while headA:
            hash_node[headA] = True
            headA = headA.next
        while headB:
            if headB in hash_node:
                return headB
            headB = headB.next
        return None


# 第二种方法:尾部相交代表在尾部对齐的情况下，同样位置遍历会遇到相同的则有相交
class Solution:
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        # 求链表长度
        tmp_headA, tmp_headB = headA, headB
        length_a, length_b, diff = 0, 0, 0
        while tmp_headA:
            length_a += 1
            tmp_headA = tmp_headA.next
        while tmp_headB:
            length_b += 1
            tmp_headB = tmp_headB.next
        # 确定谁更长谁是头
        if length_a > length_b:
            diff = length_a - length_b
            tmp_headA, tmp_headB = headA, headB
        else:
            diff = length_b - length_a
            tmp_headA, tmp_headB = headB, headA
        # 尾部对齐
        for i in range(diff):
            tmp_headA = tmp_headA.next
        # 同样位置遍历，求相同
        while tmp_headA != tmp_headB:
            tmp_headA = tmp_headA.next
            tmp_headB = tmp_headB.next
        return tmp_headA
