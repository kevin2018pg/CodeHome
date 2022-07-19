'''
Author: your name
Date: 2020-11-11 12:08:29
LastEditTime: 2020-11-12 12:54:37
LastEditors: Please set LastEditors
Description: 剑指 Offer 05. 替换空格

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
FilePath: \learningflow\Review\OfferSolution\of05_replaceSpace.py
'''


class Solution:
    def replaceSpace_01(self, s: str) -> str:
        """
        循环再合并
        """
        s_list = []
        for c in s:
            if c == ' ':
                s_list.append('%20')
            else:
                s_list.append(c)
        return ''.join(s_list)
        # return ''.join(['%20' if c == ' ' else c for c in s])

    def replaceSpace_02(self, s: str) -> str:
        """
        按' '拆分，以%20合并
        """
        return '%20'.join(s.split(' '))
