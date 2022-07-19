# -*- coding: utf-8 -*-
# @Time : 2020/6/30 21:38
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : rebotmovingCount
# @Description: 机器人运动范围（DFS/BFS）
"""逻辑和矩阵单词搜索类似，同样是矩阵搜索方法"""

"""DFS"""


class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        # 递归参数：当前元素在矩阵中的行列索引i和j ，两者的数位和si, sj
        def dfs(i, j, si, sj):
            # 终止条件：当①行列索引越界 ②数位和超出目标值k ③当前元素已访问过时，返回0 ，代表不计入可达解
            if i >= m or j >= n or si + sj > k or (i, j) in visited:
                return 0
            # 标记当前单元格：将索引(i, j)存入Set visited中，代表此单元格已被访问过
            visited.add((i, j))
            # 计算当前元素的下、右两个方向元素的数位和，并开启下层递归
            return 1 + dfs(i + 1, j, si + 1 if (i + 1) % 10 else si - 8, sj) + dfs(i, j + 1, si,
                                                                                   sj + 1 if (j + 1) % 10 else sj - 8)

        visited = set()
        return dfs(0, 0, 0, 0)


"""BFS"""


# BFS与DFS区别
# 两者目标都是遍历整个矩阵，不同点在于搜索顺序不同。DFS是朝一个方向走到底，再回退，以此类推；BFS 则是按照“平推”的方式向前搜索
# BFS实现：通常利用队列实现广度优先遍历
class Solution1:
    def movingCount(self, m: int, n: int, k: int) -> int:
        # 初始化：将机器人初始点(0,0)加入队列，
        queue, visited, = [(0, 0, 0, 0)], set()
        # 迭代终止：队列为空，代表已遍历完所有可达解
        while queue:
            # 单元格出队：将队首单元格索引与数位和弹出，作为当前搜索单元格
            i, j, si, sj = queue.pop(0)
            # 跳过条件：当①行列索引越界 ②数位和超出目标值k ③当前元素已访问过时，返回0 ，代表不计入可达解
            if i >= m or j >= n or si + sj > k or (i, j) in visited:
                continue
            # 标记当前单元格：加入set中，代表已经访问过
            visited.add((i, j))
            # 单元格入队：将当前元素的下方和右方单元格索引与数位和加入队列
            queue.append((i + 1, j, si + 1 if (i + 1) % 10 else si - 8, sj))
            queue.append((i, j + 1, si, sj + 1 if (j + 1) % 10 else sj - 8))
        # 返回队列的长度就是可达解的数量
        return len(visited)
