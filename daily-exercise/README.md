# work index & tips
1. reverse-linked-list：翻转链表（先存后转向）
2. longest-substring-without-repeating-characters：无重复最长子串（滑动窗口）
3. lru-cache：LRU缓存（哈希表+双向链表，哈希表存链表节点）
4. kth-largest-element-in-an-array：数组TopK（快速选择法，效率比快排高）
5. reverse-nodes-in-k-group：k个一组翻转链表（新增首节点，多节点记录头尾）
6. 3sum：三数之和（先排序，三指针，固定k，移动head，tail。对k，head，tail都去重）
7. 2sum：两数之和（哈希表存储数据-索引，查找target-遍历数是否存在）
8. cookies/quick_sort：快排（两种双指针，同向+内缩）
9. max-sum-subarray：最大子数组和（贪心法，状态转移，维护最大值）
10. max-dot-subarray：最大子数组乘积（乘法有正负性，维护最大和最小，更新最大和最小需考虑三个数：当前数、上一步最大积*当前数，上一步最小积*当前数）
11.merge-two-sorted-lists：合并两个有序链表（迭代法判断节点大小，移动节点，某链表结束直接接上另一链表后续节点）
12.binary-tree-level-order-traversal：二叉树层序遍历（BFS：层层加入再取值，条件是当前层存在，取出left+right赋给层变量）
13.best-time-to-buy-and-sell-stock：买卖股票最佳时机（前后最大差值，动态规划维护最低价格，遍历计算收益并维护最后返回）
14.linked-list-cycle：环形链表（遍历链表，存哈希表）
15.search-in-rotated-sorted-array：查找旋转排序数组中指定值（二分查找，始终在有序部分查找，不在再到无序再划分有序和无序）