# -*- coding: utf-8 -*-
# @Time : 2020/6/10 20:13
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0610PalindromeNumber
# @Description: 回文数

class Solution:
    # 方法一: 将int转化成str类型: 双向队列
    # 复杂度: O(n^2) [每次pop(0)都是O(n)..比较费时]
    def isPalindrome_s1(x: int) -> bool:
        res = list(str(x))
        while len(res) > 1:
            if res.pop(0) != res.pop():
                return False
        return True

    def isPalindrome_s2(x: int) -> bool:
        # 双指针法，简洁
        res = str(x)
        left, right = 0, len(res) - 1
        while left <= right:
            if res[left] != res[right]:
                return False
            left += 1
            right -= 1
        return True

    def isPalindrome_s3(x: int) -> bool:
        # 反转切片法
        return str(x) == str(x)[::-1]

    def isPalindrome_s4(self, x: int) -> bool:
        # 不需要建立列表和转字符串申请内存空间
        # 特殊情况：
        # 如上所述，当x < 0时，x不是回文数。
        # 同样地，如果数字的最后一位是0，为了使该数字为回文，
        # 则其第一位数字也应该是0
        # 只有0满足这一属性
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        revertedNumber = 0
        while x > revertedNumber:
            revertedNumber = revertedNumber * 10 + x % 10
            x //= 10
        # 当数字长度为奇数时，我们可以通过revertedNumber / 10去除处于中位的数字。
        # 例如，当输入为12321时，在while 循环的末尾我们可以得到 x = 12，revertedNumber = 123，
        # 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
        return x == revertedNumber or x == revertedNumber // 10
