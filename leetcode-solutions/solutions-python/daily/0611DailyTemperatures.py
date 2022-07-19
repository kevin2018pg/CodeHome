# -*- coding: utf-8 -*-
# @Time : 2020/6/11 19:32
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0611DailyTemperatures
# @Description: 每日温度

class Solution:
    def dailyTemperatures_s1(self, T: List[int]) -> List[int]:
        n = len(T)
        ans, nxt, big = [0] * n, dict(), 10 ** 9
        for i in range(n - 1, -1, -1):
            warmer_index = min(nxt.get(t, big) for t in range(T[i] + 1, 102))
            if warmer_index != big:
                ans[i] = warmer_index - i
            nxt[T[i]] = i
        return ans

    # 单调栈
    def dailyTemperatures_s1(self, T: List[int]) -> List[int]:
        length = len(T)
        ans = [0] * length
        stack = []
        for i in range(length):
            temperature = T[i]
            while stack and temperature > T[stack[-1]]:
                prev_index = stack.pop()
                ans[prev_index] = i - prev_index
            stack.append(i)
        return ans
