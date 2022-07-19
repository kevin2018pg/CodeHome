# -*- coding: utf-8 -*-
# @Time : 2020/7/22 20:47
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : myPow
# @Description: 数值的整数次方，实现自定义pow函数


class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        # 需要判断n小于0的情况
        if n < 0:
            return 1 / self.myPow(x, -n)
        # 判断奇偶
        if n & 1:
            # return x * self.myPow(x, n - 1)
            return x * self.myPow(x * x, n // 2)
        return self.myPow(x * x, n // 2)
