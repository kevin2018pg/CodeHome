# -*- coding: utf-8 -*-
# @Time : 2020/5/15 16:12
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0515SubarraySumEqualsK
# @Description:


def subarraySum_s1(nums, k):
    map = {0: 1}
    ans = 0
    sum = 0
    for i in range(len(nums)):
        sum = sum + nums[i]
        if sum - k in map:
            ans = ans + map[sum - k]
        if sum in map:
            map[sum] = map[sum] + 1
        else:
            map[sum] = 1
    return ans


res = subarraySum_s1([1, 1, 1], 2)
print(res)
