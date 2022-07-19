# -*- coding: utf-8 -*-
# @Time : 2020/5/8 9:28
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 03.LongestSubstringWithoutRepeatingCharacters
# @Description: 无重复字符的最长子串
"""
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度
"""

"""
双指针滑动窗口。也可以用一个set或dict替代切片来判断是否重复，速度会快一些
"""


def lengthOfLongestSubstring_s1(s):
    length = l = r = 0
    while r < len(s):
        if s[r] not in s[l:r]:
            r += 1
            length = max(length, r - l)
        else:
            l += 1
    return length


def lengthOfLongestSubstring_s2(s):
    res = 0
    mark = set()  # 用集合标明是否有出现重复字母
    r = 0  # 右指针
    for i in range(len(s)):
        if i != 0:
            mark.remove(s[i - 1])
        while r < len(s) and s[r] not in mark:  # 如果不满足条件说明r走到了s的尽头或r指向的元素
            mark.add(s[r])  # 将当前r指向的字母加入集合
            r += 1
        res = max(res, r - i)  # 在每一个位置更新最大值
    return res


num = lengthOfLongestSubstring_s2("pdvdk")
print(num)
