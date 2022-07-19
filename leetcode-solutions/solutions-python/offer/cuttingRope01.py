# -*- coding: utf-8 -*-
# @Time : 2020/7/7 20:04
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : cuttingRope01
# @Description: 减绳子题1

"""
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
题意即每段长度和等于绳子长度时，长度之积最大
"""

"""
设将长度为 n 的绳子切为 a 段：
n = n_1 + n_2 + ... + n_a
等价于求解：
max(n_1 x n_2 x ... x n_a)
"""
import math


class Solution:
    # 数学推导法：假设每一段x相等乘积最大，x=e是极值点（过程略）。由于要按整数切，所以将x=2/3代入，x=3取得最大值
    def cuttingRope_s1(self, n: int) -> int:
        if n <= 3: return n - 1  # n<4时，乘积最大是不切，题目要求段数>1，所以只切一个1出来，剩下的作为另一段
        # 三种情况，余数是0和1和2
        a, b = n // 3, n % 3
        # !!! pow()通过内置的方法直接调用，内置方法会把参数作为整型，而math模块则会把参数转换为float。
        if b == 0: return int(math.pow(3, a))  # 余数是0，直接3**a
        if b == 1: return int(math.pow(3, a - 1) * 4)  # 余数是1，直接3**a-1，拿出来一个3和余数1替换成2*2，因为4>3
        return int(math.pow(3, a) * 2)  # 余数是2，直接幂运算即可

    # 贪心算法或者也是记忆性递归
    def cuttingRope_s2(self, n: int) -> int:
        dic = {}
        dic[2] = 1
        dic[3] = 2
        dic[4] = 4
        dic[5] = 6
        dic[6] = 9

        def helper(n):
            if n in dic: return dic[n]
            res = helper(n - 3) * 3
            dic[n] = res
            return res

        return helper(n)

    # 贪心算法变形
    # TODO 此处不同于斐波那契数列，加了缓存机制反而会变慢（猜测由于数字 n 较小）
    # import functools
    # @functools.lru_cache
    def cuttingRope_s3(self, n: int) -> int:
        if n == 2: return 1
        if n == 3: return 2
        if n == 4: return 4
        if n == 5: return 6
        if n == 6: return 9
        return self.cuttingRope_s3(n - 3) * 3

    # 动态规划
    def cuttingRope_s4(self, n: int) -> int:
        # 状态定义：dp[i]表示长度为i的绳子的最大乘积值(也就表示至少切割过一次)
        # 切割点为j，j遍历1到i
        # 转移方程dp[i] = max(j段绳子最大值 * 剩下i - j段绳子最大值)
        # 其中j段绳子最大值可以分为切还是不切：max(j,dp[j]),同理有剩下i - j段绳子最大值：max(i - j, dp[i - j])
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            # 优化：这里前面切j，后面剩i - j等价于前面切i - j，后面剩j。因此j只需遍历到i // 2
            for j in range(1, i // 2 + 1):
                dp[i] = max(dp[i], max(j, dp[j]) * max(i - j, dp[i - j]))
        return dp[-1]
