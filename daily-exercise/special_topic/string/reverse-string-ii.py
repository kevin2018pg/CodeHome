# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 9:43
# @Author  : kevin
# @Version : python 3.7
# @Desc    : reverse-string-ii

class Solution:
    def reverseStr(self, s: str, k: int):
        text_array = list(s)
        for i in range(0, len(text_array), 2 * k):
            text_array[i:i + k] = self.reverse_part(text_array[i:i + k])
        return ''.join(text_array)

    def reverse_part(self, text):
        left, right = 0, len(text) - 1
        while left < right:
            text[left], text[right] = text[right], text[left]
            left += 1
            right -= 1
        return text
