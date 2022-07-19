# -*- coding: utf-8 -*-
# @Time : 2020/6/8 19:44
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0608satisfiabilityofEqualityEquations
# @Description: 等式方程的可满足性（并查集）

"""
给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。
只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 
输入：["a==b","b!=a"]
输出：false
解释：如果我们指定，a = 1 且 b = 1，那么可以满足第一个方程，但无法满足第二个方程。没有办法分配变量同时满足这两个方程。
输出：["b==a","a==b"]
输入：true
解释：我们可以指定 a = 1 且 b = 1 以满足满足这两个方程。
"""


class Solution_s1:
    def equationsPossible(self, equations: List[str]) -> bool:
        forestTree = list(range(26))

        def find(p):
            while forestTree[p] != p:
                p = forestTree[p]
            return p

        height = [1] * 26
        test = []
        for eq in equations:
            c1, c2 = ord(eq[0]) - ord('a'), ord(eq[-1]) - ord('a')
            if eq[1] == '=':
                r1 = find(c1)
                r2 = find(c2)
                if r1 == r2: continue
                # 测试数据量应该很小，路径压缩没什么提升
                if height[r1] < height[r2]:
                    forestTree[r1] = forestTree[r2]
                elif height[r2] < height[r1]:
                    forestTree[r2] = forestTree[r1]
                else:
                    forestTree[r2] = forestTree[r1]
                    height[r1] += 1
            else:
                test.append([c1, c2])
        for i, j in test:
            if find(i) == find(j): return False
        return True


class Solution_s2:
    class UnionFind():
        def __init__(self):
            # parent[0]=1, 表示0的父节点是1
            # 根节点的父节点是自己
            self.parent = list(range(26))

        # 用于寻找x的根节点
        def find(self, x):
            if x == self.parent[x]:
                return x
            # 继续向上找父节点
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        # 两个节点的合并
        def union(self, x, y):
            # x的根节点直接指向y的根节点
            self.parent[self.find(x)] = self.find(y)

    def equationsPossible(self, equations: List[str]) -> bool:
        uf = Solution_s2.UnionFind()
        for item in equations:
            if item[1] == '=':
                x = ord(item[0]) - ord('a')
                y = ord(item[3]) - ord('a')
                # 相等的进行合并操作
                uf.union(x, y)

        for item in equations:
            if item[1] == '!':
                x = ord(item[0]) - ord('a')
                y = ord(item[3]) - ord('a')
                # 判断两节点的根节点是否相同
                if uf.find(x) == uf.find(y):
                    return False

        return True


class UF:
    parent = {}

    def __init__(self, equations):
        for eq in equations:
            self.parent[eq[0]] = eq[0]
            self.parent[eq[3]] = eq[3]

    def find(self, x):
        while x != self.parent[x]:
            x = self.parent[x]
        return x

    def union(self, p, q):
        if self.connected(p, q): return
        self.parent[self.find(p)] = self.find(q)

    def connected(self, p, q):
        return self.find(p) == self.find(q)


class Solution_s3:
    def equationsPossible(self, equations: List[str]) -> bool:
        uf = UF(equations)
        for eq in equations:
            if eq[1] == '=':
                uf.union(eq[0], eq[3])
        for eq in equations:
            if eq[1] == '!' and uf.connected(eq[0], eq[3]):
                return False
        return True
