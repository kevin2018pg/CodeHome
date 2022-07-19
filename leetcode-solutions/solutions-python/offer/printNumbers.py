# -*- coding: utf-8 -*-
# @Time : 2020/7/23 8:59
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : printNumbers
# @Description:  打印从1到最大的n位数

class Solution:
    # end = 10**n - 1
    def printNumbers(self, n: int) -> List[int]:
        res = []
        for i in range(1, 10 ** n):
            res.append(i)
        return res
        # 简便写法
        # return list(range(1, 10 ** n))
