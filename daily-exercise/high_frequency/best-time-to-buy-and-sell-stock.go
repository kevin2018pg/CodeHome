//输入：[7,1,5,3,6,4]
//输出：5
//解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
//注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。如果你不能获取任何利润，返回 0(降序数组)

func maxProfit(prices []int) int {
    minPrice := 1e6   // 设置超过范围最大数，为了开启循环时使得设置数组第一个price为最小price；也可以直接设置，然后从第2个开始遍历
    maxProfit := 0  // 贪心最大利益
    for _,price := range prices {
        // 动态规划
        minPrice = min(price,minPrice)  // 更新最小入手价
        profit := price - minPrice  // 计算收益
        maxProfit = max(profit,maxProfit)   // 收益最大
    }
    return maxProfit
}
// [2,5,1,3]
func max(x,y int) int {
    if x>y {
        return x
    }
    return y
}
func min(x,y int) int {
    if x>y {
        return y
    }
    return x
}