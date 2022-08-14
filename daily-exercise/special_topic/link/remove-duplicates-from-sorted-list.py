# -*- coding: utf-8 -*-
# @Time    : 2022/8/14 18:04
# @Author  : kevin
# @Version : python 3.7
# @Desc    : remove-duplicates-from-sorted-list

# 删除重复元素
class Solution:
    def deleteDuplicates(self, head):
        if not head:
            return head

        cur = head
        while cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next

        return head
