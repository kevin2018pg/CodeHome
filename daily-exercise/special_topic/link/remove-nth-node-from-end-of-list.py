# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 15:41
# @Author  : kevin
# @Version : python 3.7
# @Desc    : remove-nth-node-from-end-of-list

# 删除倒数第n个节点
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 第一种最直观方法，先迭代链表获取长度，然后反向求删除节点的位置
class Solution:
    def removeNthFromEnd(self, head, n):
        def getLength(head):
            length = 0
            while head:
                length += 1
                head = head.next
            return length

        # 获取链表长度
        length = getLength(head)
        del_pos = length - n + 1  # 删除节点位置
        # 虚拟头，因为可能删除第一个head节点
        prehead = ListNode(next=head)
        cur = prehead
        count_pos = 0  # 位置计数
        while cur:  # *不能用cur.next，因为每一步都会更新cur节点。如果删除的是最后一个元素，cur会变为none，none没有next
            count_pos += 1
            if count_pos == del_pos:
                cur.next = cur.next.next
            cur = cur.next
        return prehead.next


# 第二种方法，双指针的巧妙思路在于，用两个指针确定出n的位置，那么两个指针之间的距离是n，一个在末尾，一个在倒数第n个的前一步，所以：
# 让快指针先走n步慢指针再开始走，这样当快指针走完的时候，慢指针下一步便是倒数第n个节点。
class SolutionDoublePoint:
    def removeNthFromEnd(self, head, n):
        # 虚拟头，因为可能删除第一个head节点
        prehead = ListNode(next=head)
        slow, fast = prehead, prehead
        count = 0  # 计数
        while fast.next != None:  # *不能用cur.next，因为每一步都会更新cur节点。如果删除的是最后一个元素，cur会变为none，none没有next
            count += 1
            if count > n:
                slow = slow.next
            fast = fast.next
        slow.next = slow.next.next

        return prehead.next


test = Solution()
a = test.removeNthFromEnd([1], 1)
