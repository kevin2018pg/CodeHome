# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 18:15
# @Author  : kevin
# @Version : python 3.7
# @Desc    : k个一组翻转链表

# 给你链表的头节点 head ，每k个节点一组进行翻转，请你返回修改后的链表。
# k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是k的整数倍，那么请将最后剩余的节点保持原有顺序。
# 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution():
    def reverse(self, head, tail):
        pre, cur = tail.next, head
        # **不能用while cur，cur到tail节点时，next不为None，又指向tail.next，死循环超时;
        # 因此这时候可以结束循环了,此时pre移动至tail，cur移动至tail.next
        while pre != tail:
            nex = cur.next
            cur.next = pre
            pre, cur = cur, nex
        return tail, head

    def reverseKGroup(self, head, k):
        pre_head = ListNode(0)  # 构造第一个sub link，用于指向head和返回（返回next即可）
        pre_head.next = head  # 指向head
        pre = pre_head  # 存储前一个sub link尾节点
        while head:
            tail = pre  # 初始化尾节点
            for i in range(k):
                tail = tail.next  # 移动尾节点k位，为空时返回
                if not tail:
                    return pre_head.next
            nex = tail.next  # 存储sub link的下一个指向
            head, tail = self.reverse(head, tail)
            # 把反转后sub link重新接回原链表
            pre.next = head
            tail.next = nex
            # 重新定义新的pre和head节点，进行下一次
            pre = tail
            head = tail.next
        return pre_head.next
