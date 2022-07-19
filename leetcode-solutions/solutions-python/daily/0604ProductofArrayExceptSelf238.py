# -*- coding: utf-8 -*-
# @Time : 2020/6/4 8:56
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0604ProductofArrayExceptSelf238
# @Description: 除自身以外数组的乘积

"""
给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
输入: [1,2,3,4]
输出: [24,12,8,6]

提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。
说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
进阶：
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
"""

"""左右数组乘积法"""
# 题目要求不能用除法，而且用数组总乘积除以下标i的数，当i是0的时候也是错误的
"""
1、初始化两个空数组 L 和 R。对于给定索引 i，L[i] 代表的是 i 左侧所有数字的乘积，R[i] 代表的是 i 右侧所有数字的乘积。
2、我们需要用两个循环来填充 L 和 R 数组的值。对于数组 L，L[0] 应该是 1，因为第一个元素的左边没有元素。对于其他元素：L[i] = L[i-1] * nums[i-1]。
3、同理，对于数组 R，R[length-1] 应为 1。length 指的是输入数组的大小。其他元素：R[i] = R[i+1] * nums[i+1]。
4、当 R 和 L 数组填充完成，我们只需要在输入数组上迭代，且索引 i 处的值为：L[i] * R[i]。
"""


def productExceptSelf_s1(nums):
    length = len(nums)
    # L 和 R 分别表示左右两侧的乘积列表
    L, R, answer = [0] * length, [0] * length, [0] * length
    # L[i] 为索引 i 左侧所有元素的乘积
    # 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
    L[0] = 1
    for i in range(1, length):
        L[i] = nums[i - 1] * L[i - 1]
    R[length - 1] = 1
    for i in reversed(range(length - 1)):
        R[i] = nums[i + 1] * R[i + 1]
    # 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
    for i in range(length):
        answer[i] = L[i] * R[i]
    return answer


"""
初始化数组长度n,初始化res=[0,0,...,0],初试化乘积k=1
从左向右遍历，遍历区间[0,n)：
res每个位置保存它左侧所有元素的乘积。即res[i]=k,k*=nums[i]

重置乘积k=1，用来保存元素右边的乘积和
从右向左遍历，遍历区间(n,0]：
res[i]*=k，表示将当前位置的左积乘以右积。
更新右积k*=nums[i]
返回res
"""


# 把左数组存放在输出数组里，右数组更新的同事乘以左数组，空间复杂度常数级
def productExceptSelf_s2(nums):
    n = len(nums)
    res = [0] * n
    k = 1
    for i in range(n):
        res[i] = k
        k = k * nums[i]
    k = 1
    for i in range(n - 1, -1, -1):
        res[i] *= k
        k *= nums[i]
    return res
