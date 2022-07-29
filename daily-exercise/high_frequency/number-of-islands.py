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
class SolutionDeep:
    def dfs(self, grid, r, c):
        grid[r][c] = 0  # 岛屿标记，代表已被访问过
        nr, nc = len(grid), len(grid[0])
        # -1 0, 1 0, 0 -1, 0 1
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid):
        width = len(grid)
        if width == 0:
            return 0
        depth = len(grid[0])

        land_num = 0
        for r in range(width):
            for c in range(depth):
                if grid[r][c] == "1":
                    land_num += 1
                    self.dfs(grid, r, c)

        return land_num


# 广度优先搜索
# 使用双端队列
import collections


class SolutionWide:
    def numIslands(self, grid):
        width = len(grid)
        if width == 0:
            return 0
        depth = len(grid[0])
        land_num = 0

        for w in range(width):
            for d in range(depth):
                if grid[w][d] == "1":
                    land_num += 1
                    grid[w][d] = 0
                    neighbors = collections.deque([(w, d)])
                    while neighbors:
                        row, col = neighbors.popleft()
                        for x, y in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                            if 0 <= x < width and 0 <= y < depth and grid[x][y] == "1":
                                neighbors.append((x, y))
                                grid[x][y] = 0
        return land_num


s = SolutionDeep()
s.numIslands(
    grid=[["1", "1", "1", "1", "0"],
          ["1", "1", "0", "1", "0"],
          ["1", "1", "0", "0", "0"],
          ["0", "0", "0", "0", "0"]])
