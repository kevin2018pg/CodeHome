# -*- coding: utf-8 -*-
# @Time : 2020/7/1 8:17
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : linkedreversePrint
# @Description: 倒序打印链表


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    """递归法"""

    # 先走至链表末端，回溯时依次将节点值加入列表，这样可以实现链表倒序输出
    def reversePrint_s1(self, head: ListNode) -> List[int]:
        # 每次传入next走到末端，以head=none（链表尾部）为递归终止条件，此时返回空列表
        # 递归回溯时每次返回当前list+当前节点值val，可实现节点倒序输出
        return self.reversePrint_s1(head.next) + [head.val] if head else []

    """辅助栈法"""

    def reversePrint_s2(self, head: ListNode) -> List[int]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]  # reverse(res)
        # pop的方法
        # stack = []
        # while head:  # push
        #     stack.append(head.val)
        #     head = head.next
        # res = []
        # while stack:  # pop
        #     res.append(stack.pop())
        # return res
