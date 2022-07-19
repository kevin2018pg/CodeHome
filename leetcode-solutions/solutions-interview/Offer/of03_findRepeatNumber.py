'''
Author: your name
Date: 2020-11-10 12:41:32
LastEditTime: 2020-11-10 23:39:08
LastEditors: Please set LastEditors
Description: 剑指 Offer 03. 找出数组中重复的数字

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 

FilePath: \learningflow\Review\OfferSolution\findRepeatNumber.py
'''


class Solution:
    def findRepeatNumber_01(self, nums: List[int]) -> int:
        """
        哈希表/set：最自然，需要额外空间
        """
        num_set = set()
        for n in nums:
            if n in num_set: return n
            num_set.add(n)
        return -1

        # num_dict = {}
        # for index, value in enumerate(nums):
        #     if value in num_dict: return value
        #     num_dict[value] = index
        # return -1

    def findRepeatNumber_02(self, nums: List[int]) -> int:
        """排序：修改原数组，O(NlogN)"""
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                return nums[i]
        return -1

    def findRepeatNumber_03(self, nums: List[int]) -> int:
        """原地交换：题意在长度为n的数组里数字范围在0-n-1之间，表示索引和值一对多"""
        # i = 0
        # while i < len(nums):
        #     # 数字已在对应索引位置，跳过
        #     if nums[i] == i:
        #         i += 1
        #         continue
        #     # 索引 nums[i] 处和索引 i 处的元素值都为 nums[i]，即重复返回 nums[i]
        #     if nums[nums[i]] == nums[i]: return nums[i]
        #     # 交换索引为 i 和 nums[i] 的元素值
        #     # TODO tips：Python中，a, b = c, d 原理是先暂存元组 (c, d)，然后 “按左右顺序” 赋值给 a 和 b，所以若nums[i] 写在前会先被赋值，之后 nums[nums[i]] 指向报错
        #     nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        # return -1

        for inx, val in enumerate(nums):
            while idx != val:
                if val == nums[val]: return val
                nums[idx], nums[val] = nums[val], nums[idx]
        return -1
