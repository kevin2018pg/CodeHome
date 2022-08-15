# work index & tips
### 链表
1. reverse-linked-list：翻转链表（先存后转向）
2. linked-list-cycle：环形链表（遍历链表，存哈希表）
3. merge-two-sorted-lists：合并两个有序链表（迭代法判断节点大小，移动节点，某链表结束直接接上另一链表后续节点）
4. reverse-nodes-in-k-group：k个一组翻转链表（新增首节点，多节点记录头尾）
5. lru-cache：LRU缓存（哈希表+双向链表，哈希表存链表节点）
### 滑动窗口/二分
1. longest-substring-without-repeating-characters：无重复最长子串（哈希表判断，滑动窗口）
2. search-in-rotated-sorted-array：查找旋转排序数组中指定值（二分查找，始终在有序部分查找，不在再到无序再划分有序和无序再寻找）
3. 2sum：两数之和（哈希表存储数据-索引，查找target-遍历数是否存在）
4. 3sum：三数之和（先排序，三指针，固定k，移动head，tail。对k，head，tail都去重）
5. merge-sorted-array：合并有序数组，类似于合并有序链表，双指针方法，倒序方法更巧妙
6. sqrtx：平方根最近整数（二分查找，mid值平方大于x，则左缩；反之更新结果值，右缩）
7. is-subsequence：判断是否是子序列（双指针滑动窗口，相等移动大小数组指针，不相等移动大数组指针）
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


# 专项练习
## DP
DP四步：
- 确定dp数组（dp table）以及下标的含义
- 确定递推公式
- dp数组如何初始化
- 确定遍历顺序
Review：
- 求次数、数值等直接状态转移，求最大最小则max/min
- 斐波那契、爬楼梯：都是前两个转移到当前，初始化单个dp数组，一层遍历，注意遍历次数
- 子序列问题：等于两个数组，需要二维dp数组或者二层遍历。概括来说：不连续递增子序列的跟前0-i 个状态有关，连续递增的子序列只跟前一个状态有关
- 多序列：多个dp数组，如机器人二维棋盘路线和两个数组最长公共子序列问题一样，一个是状态转移（注意最后字符==or!=情况），一个是求max

1. fibonacci-number：斐波那契数列（dp[0]=0,dp[1]=1,dp[i]=dp[i-1]+dp[i-2]，不需维护一个dp数组，维护两个数即可，对已知项直接返回，剩下遍历n-1次即可）
2. climbing-stairs：爬楼梯（dp[1]=1,dp[2]=2,dp[i]=dp[i-1]+dp[i-2]，遍历台阶数即可）
3. climbing-stairs-min-cost：爬楼梯最小花费（dp[0]=cost[0],dp[1]=cost[1],dp[i]=min(dp[i-1],dp[i-2])+cost[i],返回最后两个最小即可）
4. unique-paths：机器人不同路径（dp[0][j]=1,dp[i][0]=1,dp[i][j]=dp[i-1][j]+dp[i][j-1]）
5. unique-paths-ii：机器人不同路径障碍物版（dp[0][j]=0,dp[i][0]=0，障碍物之前初始化为1，dp[i][j]=dp[i-1][j]+dp[i][j-1]，前提是此点无障碍物）
6. integer-break：整数拆分最大乘积（初始化从2开始的数组(0和1拆分乘积无意义)，i从3开始，j为拆分项值从1开始，则dp[i]= max(dp[i], max(j * (i - j), j * dp[i - j])),dp[i-j]是i-j的拆分项最大乘积）
7. longest-increasing-subsequence：最大上升子序列（初始化1的dp数组，外层循环数组，内层循环到当前下标的子数组，判断内层与外层大小，满足则dp[i] = max(dp[i], dp[j] + 1)）
8. number-of-longest-increasing-subsequence：最大上升子序列个数-第7题升级版 （初始化两个数组，一个记录最大长度一个记录个数。）
9. longest-continuous-increasing-subsequence：最大连续上升子序列（当前状态只与前一个状态有关，dp[i]=dp[i-1]+1）
10.maximum-length-of-repeated-subarray：最长重复子数组（类似多序列机器人路径问题，有相等条件限制，dp[i][j]=dp[i-1][j-1]+1）
11.uncrossed-lines：不相交的线（此题等同于求最长公共子序列）
12.max-sum-subarray：最大子序列和（dp[i]=max(dp[i-1]+nums[i],nums[i])，result=max(result,dp[i])）

## array
数组：
- 顺序数组寻找元素或者近似值优先二分查找
- 单数组去重移除元素，优先快慢指针，或者头尾双指针
1. cookies/binary-search：二分查找（两个区间版本）
2. sqrtx：平方根最近整数（二分查找，mid值平方大于x，则左缩；反之更新结果值，右缩）
3. search-insert-position：查找插入位置（二分查找，找近似）
4. find-first-and-last-position-of-element-in-sorted-array：查找排序数组中元素起止位置（第3题升级版，二分查找+前后循环再查找）
5. remove-element：移除元素（快慢指针或者双向指针，覆盖元素）
6. remove-duplicates-from-sorted-array：去除有序数组重复元素（快慢指针，覆盖元素，与上面区别是从1开始遍历判断重复，前后不相等覆盖值移动慢指针）
7. move-zeroes：移动0到末尾，等同于第5第6题（快慢指针，寻找非元素值、非重复值、非0值）
8. minimum-size-subarray-sum：和大于等于指定值长度最小的子数组（快慢指针：快指针往上加和，达到目标值时，不断收缩慢指针求最小长度；区别于前面的快慢指针）
9. squares-of-a-sorted-array：升序数组平方后再升序排列（首尾指针，比较平方大小，大的加入新数组，移动一侧指针;tip:可以直接取左边负数或者双方绝对值进行比较）
10.spiral-matrix-ii：螺旋矩阵2（注意都要是左闭右开，前闭后开，注意：计算奇偶的迭代层数，初始值，以及初始值和计数更新，计数时，需填充中心点。注意首尾控制变化）
11.spiral-matrix：螺旋矩阵（螺旋2升级版，注意同样是是左闭右开，前闭后开，注意：计算行列不一致的情况如何计算迭代层数和中心起始点，更新中心行或者中心列）

## link
链表：
- 虚拟头节点很重要，常用于返回
- 排序则需线性表存储，存在则需哈希表判断，删除元素需要后移判断，翻转链表需循环改指向，局部则断开翻转
1. remove-linked-list-elements：删除链表指定值元素（迭代法，创建虚拟头，判断next值是否为指定值，删除cur.next = cur.next.next）
2. reverse-linked-list：翻转链表（前后双指针：先存后转再移动）
3. swap-nodes-in-pairs：两两交换链表节点（翻转两个局部链表节点需要用到三个节点-前中后，因为需要更新前后的指向，所以创建一个虚拟头，更新虚拟头）
4. remove-nth-node-from-end-of-list：删除链表倒数第n个节点（2方法：1、先求链表长度得到正数位置，再遍历删除；2、双指针，之间间隔为n，快针走到末尾，慢针下一个位置就是要删除元素）
5. intersection-of-two-linked-lists：相交链表（尾部相交，2方法：1哈希表，等同于判断链表是否有环，即节点是否已存在；2双指针，先确定长的链表，进行尾部对齐，开始遍历求相等节点）
6. linked-list-cycle-ii：环形链表2（和1方法一样，哈希表）
7. reverse-linked-list-ii：翻转局部链表（找到左右节点和左前右后节点，断开连接，进行局部翻转，重新建立连接）
8. reorder-list：重排链表（线性表存储，头尾访问更改指向）
9. remove-duplicates-from-sorted-list（删除重复元素，等同于第1题 删除链表指定元素）
10.remove-duplicates-from-sorted-list-ii（删除重复元素不保留，和上一题区别是需要循环删除所有值元素。两层循环）




