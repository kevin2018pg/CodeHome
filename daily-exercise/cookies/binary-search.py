# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 20:26
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 二分

# 二分查找一定是排序数组

# 左闭右闭版本，left、right完全等于索引，while循环可以相等，且right需更新为mid-1，因为<=，所以left可能会等于mid-1
class Solution_allbi:
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle - 1
            elif nums[middle] < target:
                left = middle + 1
            else:
                return nums[middle]
        return -1


'''
func search(nums []int, target int) int {
    high := len(nums)-1
    low := 0
    for low <= high {
        mid := low + (high-low)/2
        if nums[mid] == target {
            return mid
        } else if nums[mid] > target {
            high = mid-1
        } else {
            low = mid+1
        }
    }
    return -1
}
'''


# 左闭右开版本，right完全等于长度，while循环不可以相等，且right需更新为mid，因为<，所以left不会等于mid
class Solution_leftbi:
    def search(self, nums, target):
        left, right = 0, len(nums)
        while left < right:
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle
            elif nums[middle] < target:
                left = middle + 1
            else:
                return middle
        return -1
