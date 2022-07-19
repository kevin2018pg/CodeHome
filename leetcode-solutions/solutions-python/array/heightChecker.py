# -*- coding: utf-8 -*-
# @Time : 2020/5/31 21:52
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : heightChecker
# @Description: 高度检查器

"""
学校在拍年度纪念照时，一般要求学生按照 非递减 的高度顺序排列。
请你返回能让所有学生以 非递减 高度排列的最小必要移动人数。
注意，当一组学生被选中时，他们之间可以以任何可能的方式重新排序，而未被选中的学生应该保持不动。

输入：heights = [1,1,4,2,1,3]
输出：3
解释：
当前数组：[1,1,4,2,1,3]
目标数组：[1,1,1,2,3,4]
在下标 2 处（从 0 开始计数）出现 4 vs 1 ，所以我们必须移动这名学生。
在下标 4 处（从 0 开始计数）出现 1 vs 3 ，所以我们必须移动这名学生。
在下标 5 处（从 0 开始计数）出现 3 vs 4 ，所以我们必须移动这名学生。
"""


def heightChecker_s1(heights):
    target = sorted(heights)
    move = 0
    for i in range(len(heights)):
        # 记录diff的次数
        if heights[i] != target[i]:  # 推荐使用判断相等，不推荐使用减法，减法需要计算，提高时间复杂度
            move += 1
    return move


def heightChecker_s2(heights):
    target = sorted(heights)
    diff = [i for i in list(map(lambda x, y: x - y, heights, target)) if i != 0]
    return len(diff)


def heightChecker_s3(heights):
    return sum(x != y for x, y in zip(heights, sorted(heights)))
