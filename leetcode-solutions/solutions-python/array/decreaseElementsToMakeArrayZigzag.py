# -*- coding: utf-8 -*-
# @Time : 2020/5/30 15:19
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : decreaseElementsToMakeArrayZigzag
# @Description: 递减元素使数组呈锯齿状
"""
给你一个整数数组nums，每次操作会从中选择一个元素并将该元素的值减少1。
如果符合下列情况之一，则数组A就是锯齿数组：
每个偶数索引对应的元素都大于相邻的元素，即A[0] > A[1] < A[2] > A[3] < A[4] > ...
或者，每个奇数索引对应的元素都大于相邻的元素，即A[0] < A[1] > A[2] < A[3] > A[4] < ...
返回将数组nums转换为锯齿数组所需的最小操作次数
输入：nums = [1,2,3]
输出：2
解释：我们可以把 2 递减到 0，或把 3 递减到 1。
输入：nums = [9,6,1,6,2]
输出：4
"""
"""
选取起点为0或1就可以求出奇偶数组的情况了。
"""


def movesToMakeZigzag(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n, ans = len(nums), [0, 0]
    for k in [0, 1]:
        for i in range(k, n, 2):
            d = 0
            if i > 0:
                d = max(d, nums[i] - nums[i - 1] + 1)
            if i + 1 < n:
                d = max(d, nums[i] - nums[i + 1] + 1)
            ans[k] += d
    return min(ans)
