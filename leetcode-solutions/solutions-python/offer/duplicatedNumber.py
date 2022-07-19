# -*- coding: utf-8 -*-
# @Time : 2020/6/11 21:28
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : duplicatedNumber
# @Description: 找出数组中重复的数字

class Solution:
    # 哈希
    def findRepeatNumber_s1(self, nums: List[int]) -> int:
        res = {}
        for i in nums:
            if i not in res:
                res[i] = 1
            else:
                return i

    # 集合
    def findRepeatNumber_s1(self, nums: List[int]) -> int:
        repeatDict = set()
        for num in nums:
            if num in repeatDict:
                return num
            else:
                repeatDict.add(num)

    # 排序后比较相邻元素
    def findRepeatNumber_s1(self, nums: List[int]) -> int:
        nums = sorted(nums)
        num = nums[0]
        for i in range(1, len(nums)):
            if num == nums[i]:
                return num
            else:
                num = nums[i]
