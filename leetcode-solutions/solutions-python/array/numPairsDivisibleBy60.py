# -*- coding: utf-8 -*-
# @Time : 2020/5/11 23:12
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : numPairsDivisibleBy60
# @Description:

"""
在歌曲列表中，第 i 首歌曲的持续时间为 time[i] 秒。
返回其总持续时间（以秒为单位）可被 60 整除的歌曲对的数量。
形式上，我们希望索引的数字 i 和 j 满足  i < j 且有 (time[i] + time[j]) % 60 == 0。
示例：
输入：[30,20,150,100,40]
输出：3
解释：这三对的总持续时间可被 60 整数：
(time[0] = 30, time[2] = 150): 总持续时间 180
(time[1] = 20, time[3] = 100): 总持续时间 120
(time[1] = 20, time[4] = 40): 总持续时间 60

理解：
1、需要进行计数
2、若相加能被60整除：第一种情况是是60的倍数，余数都是0；第二种情况是两数除以60余数相加一定等于60。
"""


def numPairsDivisibleBy60_s1(time):
    res = 0
    hashmap = dict()
    hashmap[0] = 0
    for t in time:
        if t % 60 in hashmap:
            res += hashmap[t % 60]
        if t % 60 == 0:
            hashmap[0] += 1
            continue
        if (60 - t % 60) in hashmap:
            hashmap[60 - t % 60] += 1
        else:
            hashmap[60 - t % 60] = 1
    return res


def numPairsDivisibleBy60_s2(time):
    from collections import defaultdict  # 通过Key访问字典，Key不存在，会引发‘KeyError’。使用collections类中的defaultdict()为字典提供参数类型的默认值。
    hashmap = defaultdict(int)  # key不存在的默认值为0
    res = 0  # 记录数目
    for t in time:
        # 计数器添加次数，若余数存在于字典中，代表有相加等于60倍数的组合。
        if t % 60 in hashmap:
            res += hashmap[t % 60]
        # 第一种情况：正好被整除，直接dict[0]+=1
        if t % 60 == 0:
            hashmap[0] += 1
            continue
        # 第二种情况：不能被整除，将60与余数的差计数添加进字典（之后遇到余数存在于字典的情况代表存在他的补，计数为n，即代表存在n种组合）
        hashmap[60 - t % 60] += 1
    return res


def numPairsDivisibleBy60_s3(time):
    from collections import defaultdict  # 通过Key访问字典，Key不存在，会引发‘KeyError’。使用collections类中的defaultdict()为字典提供参数类型的默认值。
    hashmap = defaultdict(int)  # key不存在的默认值为0
    res = 0  # 记录数目
    for t in time:
        if (60 - t % 60) in hashmap:
            res += hashmap[60 - t % 60]
        if t % 60 == 0:
            res += hashmap[0]
        hashmap[t % 60] += 1
    return res


# s3的方式最快

c = numPairsDivisibleBy60_s3([60, 60, 30, 20, 20, 150, 100, 40])
print(c)
