# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 9:39
# @Author  : kevin
# @Version : python 3.7
# @Desc    : lru-cache

# 请你设计并实现一个满足LRU (最近最少使用) 缓存 约束的数据结构。
# 实现 LRUCache 类：
# LRUCache(int capacity) 以 正整数 作为容量capacity 初始化 LRU 缓存
# int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
# void put(int key, int value)如果关键字key 已经存在，则变更其数据值value ；如果不存在，则向缓存中插入该组key-value 。如果插入操作导致关键字数量超过capacity ，则应该 逐出 最久未使用的关键字。
# 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

# 输入
# ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
# [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
# 输出
# [null, null, null, 1, null, -1, null, -1, 3, 4]

# 解释
# LRUCache lRUCache = new LRUCache(2);
# lRUCache.put(1, 1); // 缓存是 {1=1}
# lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
# lRUCache.get(1);    // 返回 1
# lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
# lRUCache.get(2);    // 返回 -1 (未找到)
# lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
# lRUCache.get(1);    // 返回 -1 (未找到)
# lRUCache.get(3);    // 返回 3
# lRUCache.get(4);    // 返回 4


# 双向链表节点
class ListNode():
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None


class LRUCache():
    def __init__(self, capacity: int):
        self.capacity = capacity  # 容量
        self.hashmap = dict()  # hashmap：存储key与链表节点
        # 新建头尾节点
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化头尾指向
        self.head.next = self.tail
        self.tail.pre = self.head

    # 核心：移动节点到双链表末尾
    def move_node_to_tail(self, key):
        # 先将节点从hashmap拎出来
        node = self.hashmap[key]
        node.pre.next = node.next
        node.next.pre = node.pre
        # 再将node插入到尾节点之前，建立指向
        node.pre = self.tail.pre
        node.next = self.tail
        self.tail.pre.next = node
        self.tail.pre = node

    def get(self, key):
        # 如果hashmap有此节点，需要将节点移动到双链表末尾
        if key in self.hashmap:
            self.move_node_to_tail(key)
        node_res = self.hashmap.get(key, -1)
        if node_res == -1:
            return node_res
        else:
            return node_res.val  # 节点值

    def put(self, key, val):
        if key in self.hashmap:  # 存在直接覆盖值，移动到末尾
            self.hashmap[key].val = val
            self.move_node_to_tail(key)
        else:  # 不存在新建节点，需要提前判断是否要删除
            if len(self.hashmap) == self.capacity:
                # 去掉最久没有被访问过的节点，即头节点之后的节点
                self.hashmap.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.pre = self.head
            # 新创建一个节点插入到尾节点前
            new_node = ListNode(key, val)
            self.hashmap[key] = new_node
            # 建立新节点指向
            new_node.pre = self.tail.pre
            new_node.next = self.tail
            self.tail.pre.next = new_node
            self.tail.pre = new_node
