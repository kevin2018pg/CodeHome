'''
Author: your name
Date: 2020-11-12 12:51:30
LastEditTime: 2020-11-12 13:27:00
LastEditors: Please set LastEditors
Description: 剑指 Offer 06. 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
输入：head = [1,3,2]
输出：[2,3,1]

FilePath: \learningflow\Review\OfferSolution\of06_reversePrint.py
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def reversePrint_01(self, head: ListNode) -> List[int]:
        """
        辅助栈法：压入列表，反转
        """
        stack = []
        while head:
            stack.append(head.val)
            head = head.next

        return stack[::-1]
        # res = []
        # while stack: # 栈pop
        #     res.append(stack.pop())
        # return res

    def reversePrint_02(self, head: ListNode) -> List[int]:
        """递归"""
        # 每次传入next走到末端，以head=none（链表尾部）为递归终止条件，此时返回空列表
        # 递归回溯时依次将节点值加入列表，当前 list + 当前节点值 [head.val]合并 list，尾部返回 [], 实现节点倒序输出
        return self.reversePrint(head.next) + [head.val] if head else []
