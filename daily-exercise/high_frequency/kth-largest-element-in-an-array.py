# -*- coding: utf-8 -*-
# @Time    : 2022/3/7 19:40
# @Author  : kevin
# @File    : kth-largest-element-in-an-array.py
# @Version : python 3.6
# @Desc    : 数组中第k个最大元素

# 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
# 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
# 输入: [3,2,1,5,6,4] 和 k = 2
# 输出: 5
# 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
# 输出: 4

# 先快排，再切片
class Solution:
    def findKthLargest(self, nums, k):
        # 从大到小排序
        new_nums = self.quickSort(nums)
        return new_nums[k - 1]

    # @staticmethod
    def quickSort(self, nums):
        if len(nums) < 2:
            return nums
        # k(先进行随机交换，避免顺逆序数组极端情况)
        # rd = random.randint(nums[0], nums[len(nums) - 1])
        # nums[0], nums[rd] = nums[rd], nums[0]
        nums[0], nums[-1] = nums[-1], nums[0]
        mark = nums[0]
        smaller = [i for i in nums if i < mark]
        bigger = [i for i in nums if i > mark]
        eq = [i for i in nums if i == mark]
        # 从大到小排序
        return self.quickSort(bigger) + eq + self.quickSort(smaller)


# 快速选择法，先分治，再选定区间
import random


class Solution1:
    def findKthLargest(self, nums, k):
        def quickSelect(nums, low, high, k):  # 从小到大排序
            pivot = random.randint(low, high)  # 0-len(nums)-1中随机取一个下标，避免最坏的情况
            nums[high], nums[pivot] = nums[pivot], nums[high]  # 最右边存这个支点
            i = j = low
            while j < high:
                if nums[j] <= nums[high]:  # 找到一个比支点还小的数
                    nums[i], nums[j] = nums[j], nums[i]  # i, j相互交换
                    i += 1
                j += 1  # 每一次j都要前进一步
            nums[i], nums[j] = nums[j], nums[i]  # 把pivot放到中间来

            if k < high - i + 1:  # Topk在右边
                return quickSelect(nums, i + 1, high, k)
            elif k > high - i + 1:  # Topk在左边
                return quickSelect(nums, low, i - 1, k - (high - i + 1))
            else:
                return nums[i]

        return quickSelect(nums, 0, len(nums) - 1, k)


class Solution2():
    def findKthLargest(self, nums, k):
        def partition(arr, low, high):
            pivot = arr[low]  # 选取最左边为pivot
            left, right = low, high  # 双指针
            while left < right:
                while left < right and arr[right] >= pivot:  # 找到右边第一个<pivot的元素
                    right -= 1
                arr[left] = arr[right]  # 并将其移动到left处
                while left < right and arr[left] <= pivot:  # 找到左边第一个>pivot的元素
                    left += 1
                arr[right] = arr[left]  # 并将其移动到right处
            arr[left] = pivot  # pivot放置到中间left=right处
            return left

        def randomPartition(arr, low, high):
            pivot_idx = random.randint(low, high)  # 随机选择pivot
            arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]  # pivot放置到最左边
            return partition(arr, low, high)  # 调用partition函数

        def topKSplit(arr, low, high, k):
            # mid = partition(arr, low, high)                   # 以mid为分割点【非随机选择pivot】
            mid = randomPartition(arr, low, high)  # 以mid为分割点【随机选择pivot】
            if mid == k - 1:  # 第k小元素的下标为k-1
                return arr[mid]  # 【找到即返回】
            elif mid < k - 1:
                return topKSplit(arr, mid + 1, high, k)  # 递归对mid右侧元素进行排序
            else:
                return topKSplit(arr, low, mid - 1, k)  # 递归对mid左侧元素进行排序

        n = len(nums)
        return topKSplit(nums, 0, n - 1, n - k + 1)  # 第k大元素即为第n-k+1小元素


# test = [3, 2, 3, 1, 2, 4, 5, 5, 6]
test = [3, 2, 1, 5, 6, 4]
cls = Solution1()
num = cls.findKthLargest(test, 3)
print(num)


# 完整清晰版：第k大的数不需要转换索引，只需在partition时按从大到小划分即可
def partition(nums, left, right):
    pivot = nums[left]
    i, j = left, right
    while i < j:
        while i < j and nums[j] < pivot:
            j -= 1
        nums[i] = nums[j]
        while i < j and nums[i] >= pivot:
            i += 1
        nums[j] = nums[i]
    nums[i] = pivot
    return i


def topK_split(nums, left, right, k):  # 进行数组划分，满足条件返回
    index = partition(nums, left, right)
    if index == k - 1:
        return
    elif index < k - 1:
        topK_split(nums, index + 1, right, k)
    else:
        topK_split(nums, left, index - 1, k)


def find_topK(nums, k):  # 划分好返回索引处的数
    left, right = 0, len(nums) - 1
    topK_split(nums, left, right, k)
    return nums[k - 1]
