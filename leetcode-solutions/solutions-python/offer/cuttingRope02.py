# -*- coding: utf-8 -*-
# @Time : 2020/7/9 8:32
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : cuttingRope02
# @Description: 剪绳子题2

"""
题目同剪绳子1，答案需要取模 1e9+7（1000000007），大数越界情况下的求余问题，直接用动态规划会报错
"""
import functools


class Solution:
    # 使用缓存机制会更快
    @functools.lru_cache
    def cuttingRope(self, n: int) -> int:
        if n == 2: return 1
        if n == 3: return 2
        if n == 4: return 4
        if n == 5: return 6
        if n == 6: return 9
        return self.cuttingRope(n - 3) * 3 % (10 ** 9 + 7)

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

        return helper(n) % 1000000007
