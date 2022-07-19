# -*- coding: utf-8 -*-
# @Time : 2020/5/27 20:18
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : numOfSubarrays1343
# @Description: 给你一个整数数组 arr 和两个整数 k 和 threshold 。请你返回长度为 k 且平均值大于等于 threshold 的子数组数目。

"""
输入：arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
输出：3
解释：子数组 [2,5,5],[5,5,5] 和 [5,5,8] 的平均值分别为 4，5 和 6 。其他长度为 3 的子数组的平均值都小于 4 （threshold 的值)。

输入：arr = [1,1,1,1,1], k = 1, threshold = 0
输出：5
输入：arr = [7,7,7,7,7,7,7], k = 7, threshold = 7
输出：1
输入：arr = [11,13,17,23,29,31,7,5,2,3], k = 3, threshold = 5
输出：6
解释：前 6 个长度为 3 的子数组平均值都大于 5 。注意平均值不是整数。
"""

"""
第一种方法思路：
1、首先遍历数组，将累计和存储到一个新的数组中（注意该数组的长度应该比原数组多1，初始元素0）
2、构建长度为k的滑动窗口，窗口的左右端口差值便是中间k个数字的和，取平均后与threshold做判断即可
3、注意子数组顺序与原数组一致
"""


def numOfSubarrays_s1(arr, k, threshold):
    """
    :type arr: List[int]
    :type k: int
    :type threshold: int
    :rtype: int
    """
    res = [0]
    ans = 0
    target = k * threshold
    for i in range(len(arr)):
        res.append(res[-1] + arr[i])
    for i in range(len(res) - k):
        a = res[i]
        b = res[i + k]
        if (b - a) >= target:
            ans += 1

    return ans


"""
第二种方法：一次遍历， 维护一个定长的滑动窗口
1、当 i < k - 1时（最前面的部分），只是计算窗口内数字的总和
2、当 i >= k - 1时（k个元素），滑动窗口成型，每往前走一个，判断总和是否大于等于k * threshold
3、判断完成后，去除窗口最左边的值，即arr[i - (k - 1)]
"""


def numOfSubarrays_s2(arr, k, threshold):
    ans = 0
    res = 0
    target = k * threshold
    for i in range(len(arr)):
        res += arr[i]
        if i >= k - 1:
            if res >= target:
                ans += 1
            res -= arr[i - (k - 1)]
    return ans
