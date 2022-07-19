# -*- coding: utf-8 -*-
# @Time : 2020/5/14 20:56
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : canThreePartsEqualSum
# @Description:将数组分成和相等的三个部分
"""
给你一个整数数组 A，只有可以将其划分为三个和相等的非空部分时才返回 true，否则返回 false。
形式上，如果可以找出索引 i+1 < j 且满足 (A[0] + A[1] + ... + A[i] == A[i+1] + A[i+2] + ... + A[j-1] == A[j] + A[j-1] + ... + A[A.length - 1]) 就可以将数组三等分。

示例 1：
输入：[0,2,1,-6,6,-7,9,1,2,0,1]
输出：true
解释：0 + 2 + 1 = -6 + 6 - 7 + 9 + 1 = 2 + 0 + 1
示例 2：
输入：[0,2,1,-6,6,7,9,-1,2,0,1]
输出：false
示例 3：
输入：[3,3,6,5,-2,2,5,1,-9,4]
输出：true
解释：3 + 3 = 6 = 5 - 2 + 2 + 5 + 1 - 9 + 4
"""


def canThreePartsEqualSum_s1(A):
    amount = sum(A)
    if amount % 3 != 0:  # 和不为3倍数一定不能等分
        return False
    res = amount // 3
    temp, count = 0, 0
    for i in A:
        if count == 2:  # 前两段都符合条件，直接返回
            return True  # 这段代码放这是防止[1,-1,1,-1]这种两段符合但没有第三段，如果两段符合遍历完直接返回False
        temp += i  # 加和
        if temp == res:
            count += 1  # 满足条件计数+1，和为0
            temp = 0
    # [1,-1,1,-1]这种计数满足，但是遍历完了，直接返回false
    return False


def canThreePartsEqualSum_s2(A):
    # 双指针法，计算两端值，中间不管。注意那种两端计算相同，列表的元素已经被遍历完，或者列表和等于0的情况
    amount = sum(A)
    if amount % 3 != 0:
        return False
    target = amount // 3
    left = 0
    right = len(A) - 1
    lans, rans = A[left], A[right]
    while left + 1 < right:
        if lans == target and rans == target:
            return True
        if lans != target:
            left += 1
            lans += A[left]
        if rans != target:
            right -= 1
            rans += A[right]
    return False


# res = canThreePartsEqualSum_s3([3, 3, 6, 5, -2, 2, 5, 1, -9, 4])
res = canThreePartsEqualSum_s2([1, -1, 1, -1])
print(res)
