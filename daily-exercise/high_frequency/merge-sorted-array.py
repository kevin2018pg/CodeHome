# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 20:28
# @Author  : kevin
# @Version : python 3.7
# @Desc    : merge-sorted-array
# 合并两个数组

# 1、直接合并后进行排序

# 双指针：再赋值。其中一个指针结束，把另一个数组剩余直接加进来
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        sorted = []
        up, down = 0, 0
        while up < m or down < n:
            if up == m:
                sorted.append(nums2[down])
                down += 1
            elif down == n:
                sorted.append(nums1[up])
                up += 1
            elif nums1[up] < nums2[down]:
                sorted.append(nums1[up])
                up += 1
            else:
                sorted.append(nums2[down])
                down += 1
        nums1[:] = sorted


# 3、倒序放入，先找出大的放到最后一位
class SolutionReverse:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2 = m - 1, n - 1
        tail = m + n - 1
        while p1 >= 0 or p2 >= 0:
            if p1 == -1:
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif p2 == -1:
                nums1[tail] = nums1[p1]
                p1 -= 1
            elif nums1[p1] > nums2[p2]:
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                nums1[tail] = nums2[p2]
                p2 -= 1
            tail -= 1
