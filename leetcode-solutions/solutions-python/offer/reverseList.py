# -*- coding: utf-8 -*-
# @Time : 2020/7/6 20:13
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : reverseList
# @Description: 反转链表

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 第一种方法：双指针
    """保存前一个节点 pre 和当前节点 cur, 并记录 cur 的下一个节点 nex,
      每次都将当前节点指向前一个节点, 然后 pre 和 cur 都往后移动一位即可 (即 pre = cur, cur = nex)
      注意边界条件: 没有节点的情况, 以及对 head 的 next 的处理"""

    def reverseList_s1(self, head: ListNode) -> ListNode:
        # 方法1: 迭代, 双指针
        if not head:
            # 链表为空的话直接返回空
            return head
        pre, cur = head, head.next
        # 首先head变成了翻转后的末尾, 所以其next要置为空
        head.next = None
        while cur:
            # 先存下cur的下一个节点
            nex = cur.next
            # cur的下一个节点指向pre, 完成当前节点指向的反转
            cur.next = pre
            # 更新pre和cur, 分别按照原链表顺序往后移动一位
            pre, cur = cur, nex
        # 最终cur就是空, 而pre则是反转后的开头节点
        return pre

    # 看着更加友好
    def reverseList_s11(self, head: ListNode) -> ListNode:
        # 方法1: 迭代, 双指针
        if not head:
            # 链表为空的话直接返回空
            return head
        pre, cur = None, head
        while cur:
            # 先存下cur的下一个节点
            nex = cur.next
            # cur的下一个节点指向pre, 完成当前节点指向的反转
            cur.next = pre
            # 更新pre和cur, 分别按照原链表顺序往后移动一位
            pre, cur = cur, nex
        # 最终cur就是空, 而pre则是反转后的开头节点
        return pre

    # 双指针2
    def reverseList_s2(self, head: ListNode) -> ListNode:
        if not head:
            return head
        pre = None
        cur = head
        while cur:
            cur.next, pre, cur = pre, cur, cur.next
        return pre

    # 第二种方法：递归
    def reverseList_s3(self, head: ListNode) -> ListNode:
        def helper(last, cur):
            if cur == None: return cur
            next = cur.next
            cur.next = last
            if next == None: return cur
            return helper(cur, next)

        return helper(None, head)
