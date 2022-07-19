# -*- coding: utf-8 -*-
# @Time : 2020/6/30 9:25
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : replaceSpace
# @Description: 替换空格

class Solution:
    def replaceSpace_s1(self, s: str) -> str:
        res = []
        for c in s:
            if c == ' ':
                res.append("%20")
            else:
                res.append(c)
        return "".join(res)

    def replaceSpace_s2(self, s: str) -> str:
        return ''.join(('%20' if c == ' ' else c for c in s))

    # 最快的方法
    def replaceSpace_s3(self, s: str) -> str:
        return '%20'.join(s.split(' '))
