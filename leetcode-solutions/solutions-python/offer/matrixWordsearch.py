# -*- coding: utf-8 -*-
# @Time : 2020/6/28 8:30
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : MatrixWordsearch
# @Description: 矩阵中的路径(DFS+剪枝)

"""
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]
但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
"""
"""
算法原理：
深度优先搜索： 可以理解为暴力法遍历矩阵中所有字符串可能性。DFS 通过递归，先朝一个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。
剪枝：在搜索中，遇到这条路不可能和目标字符串匹配成功的情况（例如：此矩阵元素和目标字符不同、此元素已被访问），则应立即返回，称之为 可行性剪枝 。
"""


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # 当前元素在矩阵board中的行列索引i和j ，当前目标字符在word中的索引k
        def dfs(i, j, k):
            # 递归终止条件：1、行或列索引越界；2、当前矩阵元素与目标字符不同；3、当前矩阵元素已访问过
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]: return False
            # 字符串全部匹配，返回true
            if k == len(word) - 1: return True
            # 将board[i][j]变量存于tmp，修改为字符‘/’，代表此元素已访问过，避免搜索重复访问
            tmp, board[i][j] = board[i][j], '/'
            # 朝当前元素的上、下、左、右四个方向开启下层递归，使用或连接 （代表只需一条可行路径） ，并记录结果至res
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            # 还原当前矩阵元素
            board[i][j] = tmp
            # 回溯返回值，返回结果，代表是否能搜索到目标字符串
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False
