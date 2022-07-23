# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 15:39
# @Author  : kevin
# @Version : python 3.7
# @Desc    : best-time-to-buy-and-sell-stock

# 给定一个数组 prices ，它的第i 个元素prices[i] 表示一支给定股票第 i 天的价格。
# 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
# 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0


class Solution:
    def maxProfit(self, prices):
        max_profit = 0
        min_price = 1e6
        for price in prices:
            min_price = min(min_price, price)  # DP维护最低入手价
            max_profit = max(max_profit, price - min_price)  # 计算最大收益，更新
        return max_profit
