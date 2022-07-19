# -*- coding: utf-8 -*-
# @Time : 2020/6/5 19:06
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : twoListQueue
# @Description: 两个栈实现一个队列

"""
设计栈 A 用于加入队尾操作，栈 B 用于将元素倒序，从而实现删除队首元素。
加入队尾 appendTail()函数： 将数字 val 加入栈 A 即可。
删除队首deleteHead()函数： 有以下三种情况。
    当栈 B 不为空： B中仍有已完成倒序的元素，因此直接返回 B 的栈顶元素
    否则，当 A 为空： 即两个栈都为空，无元素，因此返回 -1
    否则： 将栈 A 元素全部转移至栈 B 中，实现元素倒序，并返回栈 B 的栈顶元素
"""


class CQueue:
    def __init__(self):
        self.A, self.B = [], []

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        if self.B:
            return self.B.pop()
        if not self.A:
            return -1
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()

# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
