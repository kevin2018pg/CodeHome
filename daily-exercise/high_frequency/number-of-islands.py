# -*- coding: utf-8 -*-
# @Time    : 2022/3/12 18:12
# @Author  : kevin
# @File    : number-of-islands.py
# @Version : python 3.6
# @Desc    : 岛屿数量


# 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
# 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
# 此外，你可以假设该网格的四条边均被水包围。
# 输入：grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# 输出：1

# 深度优先搜索
class Solution:
    def dfs(self, grid, r, c):
        grid[r][c] = 0  # 岛屿标记，代表已被访问过
        nr, nc = len(grid), len(grid[0])
        # -1 0, 1 0, 0 -1, 0 1
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid):
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        land_num = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    land_num += 1
                    self.dfs(grid, r, c)

        return land_num


s = Solution()
s.numIslands(
    grid=[["1", "1", "1", "1", "0"],
          ["1", "1", "0", "1", "0"],
          ["1", "1", "0", "0", "0"],
          ["0", "0", "0", "0", "0"]])
