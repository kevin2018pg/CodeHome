# -*- coding: utf-8 -*-
# @Time : 2020/6/1 16:08
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : UniquePaths
# @Description: 不同路径(动态规划题)

"""
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
问总共有多少条不同的路径？
输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
"""


class Solution(object):
    def uniquePaths_s1(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 采用二维数组形式的动态规划
        # dp[i][j]表示当前移动到的总次数，要求f[-1][-1]
        # 状态转移公式：dp[i][j] = dp[i-1][j]+dp[i][j-1]
        # 初始值：每一步移动的次数可以看做横轴和纵轴的和，因此 dp[i][0] = 1,dp[0][j]=1
        # 运动的轨迹：要么往下，要么往左
        # 时间复杂度：O(m*n),空间复杂度：O(m*n)
        dp = [[0] * n for _ in range(m)]  # [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePaths_s2(self, m, n):
        """优化空间复杂度为O(n)"""
        # 对二维矩阵进行压缩成一位数组,将最新生成的值覆盖掉旧的值,逐行求解当前位置的最新路径条数！
        # 实质：在于动态计算并替换当前位置下的路径数最新值
        # 状态转移公式变成：dp[i] = dp[i-1]+dp[i]
        # 初始值： dp = [1]*m,取横轴
        # dp[-1]表示可能路径的总数
        # 空间复杂度：O(n),时间复杂度:O(m*n)
        dp = [1] * m
        for j in range(1, n):
            for i in range(1, m):
                dp[i] = dp[i - 1] + dp[i]
        return dp[-1]

    def uniquePaths_s3(self, m, n):
        """排列组合"""

        # ans = ((m - 1 + n - 1))! / (m - 1)!(n - 1)!
        def factor(num):
            if num < 2:
                return 1
            res = 1
            for i in range(1, num + 1):
                res *= i
            return res

        ans = factor(m + n - 2) // (factor(m - 1) * factor(n - 1))
        return ans


"""
1.定义状态：即定义数据元素的含义，这里定义dp[i][j]为当前位置的路径数，i表示i列，j表示j行
2.建立状态转移方程：因为从题目要求中可以看出，机器人只能向右或向下移动。所以到达dp[i][j]就可能是经过dp[i-1][j]到达，
也可能是经过dp[i][j-1]到达。所以状态转移方程为：dp[i][j]=dp[i-1][j]+dp[i][j-1]
3.设定初始值：通过状态转移方程可以看出，i和j下表要从1开始，否则会导致数组溢出异常。同时每一个位置点代表到达当前位置的路径条数，所以要设置最初的路径条数即dp[i][0]=1,dp[0][j]=1，即第一行，第一列值为1。
4.状态压缩：即优化数组空间，每次状态的更新只依赖于前一次的状态，优化一般作用于多维数组，观察是否可以将多维数组以一维数组来动态表示，即用一维数组来保存上次的状态。这道题的优化方法是存在的。具体看下面的代码解释。状态转移方程：dp[i] = dp[i-1] + dp[i]
5.选出结果：根据状态转移方程，求路径的总数，因此dp[-1][-1]表示的是到达最后位置的路径总条数。
"""
