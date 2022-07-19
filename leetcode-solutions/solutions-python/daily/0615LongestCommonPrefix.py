# -*- coding: utf-8 -*-
# @Time : 2020/6/15 8:23
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0615LongestCommonPrefix
# @Description: 最长公共前缀

"""
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。
输入: ["flower","flow","flight"]
输出: "fl"
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
"""


class Solution:
    """
    横向扫描法：
    LCP(S1…Sn) 表示字符串 S_1…Sn的最长公共前缀。
    可以得到以下结论：
    LCP(S1…Sn)=LCP(LCP(LCP(S1,S2),S3),…Sn)
    基于该结论，可以得到一种查找字符串数组中的最长公共前缀的简单方法。依次遍历字符串数组中的每个字符串，对于每个遍历到的字符串，更新最长公共前缀，当遍历完所有的字符串以后，即可得到字符串数组中的最长公共前缀。
    """

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        prefix, length = strs[0], len(strs)
        for i in range(1, length):
            prefix = self.lcp(prefix, strs[i])
            if not prefix:
                break
        return prefix

    def lcp(self, str1, str2):
        minlength, index = min(len(str1), len(str2)), 0
        while index < minlength and str1[index] == str2[index]:
            index += 1
        return str1[:index]

    """
    纵向比较，比较每一列，遇到长度相等或者列所在字母不相等即可返回
    第一个单次如果是最短的且都是共同，循环完返回第一个单次；若不是最短的，返回的是切片形式
    """

    def longestCommonPrefix_s2(self, strs: List[str]) -> str:
        if not strs:
            return ""
        for i in range(len(strs[0])):
            c = strs[0][i]
            for j in range(1, len(strs)):
                if i == len(strs[j]) or strs[j][i] != c:
                    return strs[0][0:i]
        return strs[0]

    """
    二分查找法
    最短路径的二分，中间值，不断更新
    """

    def longestCommonPrefix_s3(self, strs: List[str]) -> str:
        if not strs:
            return ""

        def isCommonPrefix(length):
            str0, count = strs[0][:length], len(strs)
            for i in range(1, count):
                if str0 != strs[i][:length]:
                    return False
            return True

        minlength = min(len(s) for s in strs)
        low, high = 0, minlength
        while low < high:
            mid = (high - low + 1) // 2 + low
            if isCommonPrefix(mid):
                low = mid
            else:
                high = mid - 1
        return strs[0][:low]
