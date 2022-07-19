# -*- coding: utf-8 -*-
# @Time : 2020/7/21 9:04
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : hammingWeight
# @Description: 二进制数中1的个数

# 请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。
# 例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2

class Solution:
    def hammingWeight_s1(self, n: int) -> int:
        # 位运算，逐位判断
        # n & 1 = 0，则n二进制最后一位为0
        # n & 1 = 1，则n二进制最后一位为1
        res = 0
        while n:
            res += n & 1  # n % 2
            # 将n右移
            n >>= 1  # n /= 2
        return res

    def hammingWeight_s2(self, n: int) -> int:
        # 转成二进制数（返回的是str，对'1'计数）
        return bin(n).count('1')

    def hammingWeight_s3(self, n: int) -> int:
        # 求和
        return sum(map(int, list(bin(n)[2:])))
