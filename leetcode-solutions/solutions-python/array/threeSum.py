# -*- coding: utf-8 -*-
# @Time : 2020/5/29 8:19
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : threeSum
# @Description: 三数之和
"""
给你一个包含n个整数的数组nums，判断nums中是否存在三个元素a，b，c ，使得a + b + c = 0
请你找出所有满足条件且不重复的三元组。
"""
"""
题解：排序+双指针
双指针法铺垫：先将给定 nums 排序，复杂度为 O(NlogN)
双指针法思路：固定3个指针中最左（最小）数字的指针k，双指针i，j分设在数组索引(k+1, len(nums)-1)两端，通过双指针交替向中间移动，
    记录对于每个固定指针k的所有满足nums[k]+nums[i]+nums[j] == 0的 i,j 组合：
当 nums[k] > 0 时直接break跳出：因为 nums[j] >= nums[i] >= nums[k] > 0，即 3个数字都大于0 ，在此固定指针k之后不可能再找到结果了。
当 k > 0且nums[k] == nums[k - 1]时即跳过此元素nums[k]：因为已经将 nums[k - 1] 的所有组合加入到结果中，本次双指针搜索只会得到重复组合。
i，j 分设在数组索引 (k+1, len(nums)-1)两端，当i < j时循环计算s = nums[k] + nums[i] + nums[j]，并按照以下规则执行双指针移动：
当s < 0时，i += 1并跳过所有重复的nums[i]；
当s > 0时，j -= 1并跳过所有重复的nums[j]；
当s == 0时，记录组合[k, i, j]至res，执行i += 1和j -= 1并跳过所有重复的nums[i]和nums[j]，防止记录到重复组合
"""


def threeSum(nums):
    if (not nums or len(nums) < 3):
        return []
    nums.sort()
    res, k = [], 0
    for k in range(len(nums) - 2):
        if nums[k] > 0:
            break
        if k > 0 and nums[k] == nums[k - 1]:
            continue
        i, j = k + 1, len(nums) - 1
        while i < j:
            s = nums[k] + nums[i] + nums[j]
            if s < 0:
                i += 1
                while i < j and nums[i] == nums[i - 1]: i += 1
            elif s > 0:
                j -= 1
                while i < j and nums[j] == nums[j + 1]: j -= 1
            else:
                res.append([nums[k], nums[i], nums[j]])
                i += 1
                j -= 1
                while i < j and nums[i] == nums[i - 1]: i += 1
                while i < j and nums[j] == nums[j + 1]: j -= 1
    return res
