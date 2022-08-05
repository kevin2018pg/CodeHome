# work index & tips
### 链表
1. reverse-linked-list：翻转链表（先存后转向）
2. linked-list-cycle：环形链表（遍历链表，存哈希表）
3. merge-two-sorted-lists：合并两个有序链表（迭代法判断节点大小，移动节点，某链表结束直接接上另一链表后续节点）
4. reverse-nodes-in-k-group：k个一组翻转链表（新增首节点，多节点记录头尾）
5. lru-cache：LRU缓存（哈希表+双向链表，哈希表存链表节点）
### 滑动窗口/二分
1. longest-substring-without-repeating-characters：无重复最长子串（滑动窗口）
2. search-in-rotated-sorted-array：查找旋转排序数组中指定值（二分查找，始终在有序部分查找，不在再到无序再划分有序和无序再寻找）
3. 2sum：两数之和（哈希表存储数据-索引，查找target-遍历数是否存在）
4. 3sum：三数之和（先排序，三指针，固定k，移动head，tail。对k，head，tail都去重）
5. merge-sorted-array：合并有序数组，类似于合并有序链表，双指针方法，倒序方法更巧妙
6. sqrtx：平方根最近整数（二分查找，mid值平方大于x，则左缩；反之更新结果值，右缩）
### 快排/快速选择
1. cookies/quick_sort：快排（两种双指针，同向+内缩）
2. kth-largest-element-in-an-array：数组TopK（快速选择法，效率比快排高）
### 贪心
1. max-sum-subarray：最大子数组和（贪心法，状态转移，维护最大值）
2. max-dot-subarray：最大子数组乘积（乘法有正负性，维护最大和最小，更新最大和最小需考虑三个数：当前数、上一步最大积*当前数，上一步最小积*当前数）
3. best-time-to-buy-and-sell-stock：买卖股票最佳时机（前后最大差值，动态规划维护最低价格，遍历计算收益并维护最后返回）
### 树
1. binary-tree-level-order-traversal：二叉树层序遍历（BFS：层层加入再取值，条件是当前层存在，取出left+right赋给层变量）
2. binary-tree-zigzag-level-order-traversal：二叉树层序遍历分奇偶层遍历顺序不同（同上题一样bfs，奇数层需要从右往左即是翻转层值）
3. lowest-common-ancestor-of-a-binary-tree：二叉树的最近公共祖先（递归二叉树先序遍历）
### 栈
1. valid-parentheses：有效括号（建立括号哈希表，遍历字符串，属于哈希表的key入栈，不属于则需要闭合，或者闭合完需要出栈）
### DFS/BFS
1. number-of-islands：岛屿数量（深度优先搜索，访问过的点做标记）
### DP
单个数组或字符串用一维dp，dp[i]定义为nums[0:i]中想要求的结果；两个数组或字符串用二维dp，定义成两维的dp[i][j]，其含义是在A[0:i]与B[0:j]之间匹配得到结果。
1. longest-increasing-subsequence：最长上升子序列（dp：初始状态为数组长度的1-最短是1，双层遍历，满足条件更新状态值），一维dp
2. longest-common-subsequence：最长公共子序列（dp：初始状态是二维的0，最后一位字符相等则序列+1，不相等则分别进一位取max），二维dp

