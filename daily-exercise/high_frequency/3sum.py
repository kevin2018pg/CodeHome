# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 9:33
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 三数之和(先排序再双指针)

class Solution:
    def threeSum(self, nums):
        if (not nums or len(nums) < 3):
            return []
        ## 快排
        # def quick_sort(nums, left, right):
        #     if left >= right:
        #         return
        #     可加入随机流程避免最坏情况（完全顺序或逆序）
        #     low, high = left, right
        #     pivot = nums[low]
        #     while left < right:
        #         while left < right and nums[right] > pivot:
        #             right -= 1
        #         nums[left] = nums[right]
        #         while left < right and nums[left] <= pivot:
        #             left += 1
        #         nums[right] = nums[left]
        #     nums[left] = pivot
        #     quick_sort(nums, low, left - 1)
        #     quick_sort(nums, left + 1, high)

        # left, right = 0, len(nums) - 1
        # quick_sort(nums, left, right)
        nums.sort()

        ## 三指针
        target_array = []
        for point in range(0, len(nums) - 2):  # 外层point移动
            if nums[point] > 0:  # point小于head和tail，>0表示没有符合条件的三元组
                break
            if point > 0 and nums[point] == nums[point - 1]:  # 去除point重复情况
                continue
            head, tail = point + 1, len(nums) - 1  # 确定point后，内层定义头尾双指针
            while head < tail:  # 内层双指针循环
                num_sum = nums[point] + nums[head] + nums[tail]  # sum
                if num_sum < 0:  # 移动head
                    head += 1
                    while head < tail and nums[head] == nums[head - 1]:  # 去除head重复情况
                        head += 1
                elif num_sum > 0:  # 移动tail
                    tail -= 1
                    while head < tail and nums[tail] == nums[tail + 1]:  # 去除tail重复情况
                        tail -= 1
                else:  # 符合条件，双向内缩移动
                    target_array.append([nums[point], nums[head], nums[tail]])
                    head += 1
                    tail -= 1
                    # 去除head和tail重复情况
                    while head < tail and nums[head] == nums[head - 1]:
                        head += 1
                    while head < tail and nums[tail] == nums[tail + 1]:
                        tail -= 1

        return target_array

# [-1,0,1,2,-1,-4]
