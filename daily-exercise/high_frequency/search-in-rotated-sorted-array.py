# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 21:33
# @Author  : kevin
# @Version : python 3.7
# @Desc    : search-in-rotated-sorted-array

# 旋转数组找目标
# [1,2,3,4,5,6,7]->[4,5,6,7,0,1,2]，找出0的下标位置
class Solution:
    def search(self, nums, target):
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[left]:  # 左边有序数组
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
