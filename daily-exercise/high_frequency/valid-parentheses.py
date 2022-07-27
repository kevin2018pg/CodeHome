# -*- coding: utf-8 -*-
# @Time    : 2022/7/27 20:58
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 有效的括号

# 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
# ()[]{} {()} (] ([)] ){  )) (())
# ( (
class Solution:
    def isValid(self, s):
        if len(s) % 2 == 1:  # 字符串长度必须为偶数
            return False

        stack = []  # 栈结构
        parentheses = {'(': ')', '{': '}', '[': ']'}  # 括号关系map
        for item in s:
            if item in parentheses.keys():  # 如果相同方向括号必须加进来
                stack.append(parentheses[item])
            else:
                if not stack or stack[-1] != item:  # 栈为空（都闭合了）首个括号就逆向 或者 最新加入也就是栈顶的括号不等于当前元素
                    return False
                else:  # 闭合一个之后立即弹出栈顶括号
                    stack.pop()
        return True if not stack else False  # 两边重复括号完全一致(())/()()


s = Solution()
bo = s.isValid("()()")
