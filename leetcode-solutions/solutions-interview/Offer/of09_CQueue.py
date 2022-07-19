'''
Author: your name
Date: 2020-11-18 08:23:12
LastEditTime: 2020-11-18 09:31:32
LastEditors: Please set LastEditors
Description: 剑指 Offer 09. 用两个栈实现队列
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]

FilePath: \learningflow\Review\OfferSolution\of09_CQueue.py
'''


class CQueue:
    def __init__(self):
        # 维护两个栈，初始化为空。A用于加入值，B用于倒序存储值，AB一体，B作用是倒序删除
        self.A, self.B = [], []

    # func：加入队尾
    def appendTail(self, value: int) -> None:
        self.A.append(value)

    # func：删除队首
    def deleteHead(self) -> int:
        # 先判断B：若B中有元素，即有倒序存储的值，弹出栈顶即可
        if self.B: return self.B.pop()
        # A为空，即B也为空，返回-1
        if not self.A: return -1
        # A中有元素，全部弹出加入B，实现倒序存储
        while self.A:
            self.B.append(self.A.pop())
        # 弹出B栈顶
        return self.B.pop()

    # 第二种写法
    # def deleteHead(self) -> int:
    #     if not self.B:    # B为空
    #         if not self.A:  # A B都为空
    #             return -1
    #         else:  # 把A栈中的东西全部倒入B栈中，倒序存储
    #             while self.A:
    #                 self.B.append(self.A.pop())
    #     return self.B.pop()   # B不为空，弹出栈顶

# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
