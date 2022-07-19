# -*- coding: utf-8 -*-
# @Time : 2020/7/23 20:24
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : deleteNode
# @Description: 删除链表的节点

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 双指针后移
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        # 头结点等于val，直接返回head.next
        if head.val == val: return head.next
        # 节点初始化
        pre, cur = head, head.next
        # 当cur节点为空或者cur节点值等于val跳出（为空则说明链表中没有val）
        while cur and cur.val != val:
            # 往后移动
            pre, cur = cur, cur.next
        # 执行删除操作，前节点指向当前节点下一个节点
        if cur: pre.next = cur.next
        # 返回链表头部节点head
        return head

    # 伪节点单指针循环判断
    def deleteNode_s1(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(0)  # 设置伪结点
        dummy.next = head
        if head.val == val: return head.next  # 头结点是要删除的点，直接返回
        while head and head.next:
            if head.next.val == val:  # 下个节点的节点值
                head.next = head.next.next
            head = head.next
        return dummy.next
