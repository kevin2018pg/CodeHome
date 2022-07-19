# -*- coding: utf-8 -*-
# @Time : 2020/6/9 20:07
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0609translateNum
# @Description: 把数字翻译成字符串（动态规划）

"""
给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
"""


# 解题思路：
# 根据题意，总结出 “递推公式” （即转移方程）。此题可用动态规划解决，以下按照流程解题。
# 状态定义： 设动态规划列表 dp，dp[i] 代表以 xi为结尾的数字的翻译方案数量。
# 转移方程： 若 xi和 x{i-1}组成的两位数字可以被翻译，
#     则dp[i]=dp[i−1]+dp[i−2] ；否则dp[i]=dp[i−1]
# 初始状态：dp[0]=dp[1]=1 ，即 “无数字” 和 “第 1 位数字” 的翻译方法数量均为 1；
# 返回值：dp[n] ，即此数字的翻译方案数量。
# Q： 无数字情况dp[0]=1 从何而来？
# A： 当num 第1,2 位的组成的数字 in [10,25]时，显然应有 22 种翻译方法，即dp[2]=dp[1]+dp[0]=2 ，而显然 dp[1] = 1，因此推出dp[0]=1 。


class Solution_s1:
    def translateNum(self, num: int) -> int:
        a, b = 1, 1
        s = str(num)
        for i in range(2, len(s) + 1):
            # tmp = s[i - 2:i]
            # c = a + b if "10" <= tmp <= "25" else a
            # b = a
            # a = c
            a, b = (a + b if "10" <= s[i - 2:i] <= "25" else a), a
        return a


# 由于动态规划计算是对称的，即从左向右遍历（从第 dp[2]计算至dp[n] ）和从右向左遍历（从第p[n−2]计算至dp[0]）所得方案数一致
class Solution_s2:
    def translateNum(self, num: int) -> int:
        a, b = 1, 1
        s = str(num)
        for i in range(len(s) - 2, -1, -1):
            # tmp = s[i:i + 2]
            # c = a + b if "10" <= tmp <= "25" else a
            # b = a
            # a = c
            a, b = (a + b if "10" <= s[i:i + 2] <= "25" else a), a
        return a
