// 最大子数组和
// 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
// 子数组 是数组中的一个连续部分。
// solution
// 1, dp
// 2, 贪心-状态传递：f(i)=max{f(i−1)+nums[i],nums[i]}
func maxSubArray(nums []int) int {
    maxSum := nums[0]
    for i:= 1;i<len(nums);i++ {
        if nums[i] + nums[i-1] > nums[i] {  // 加和变大
            nums[i] += nums[i-1]    // 状态转移
        }
        if nums[i] > maxSum {   // 更新sub和
            maxSum = nums[i]
        }
    }
    return maxSum
}



func maxSubArrayDp(nums []int) int {
    n := len(nums)
    // 这里的dp[i] 表示，最大的连续子数组和，包含num[i] 元素
    dp := make([]int,n)
    // 初始化，由于dp 状态转移方程依赖dp[0]
    dp[0] = nums[0]
    // 初始化最大的和
    mx := nums[0]
    for i:=1;i<n;i++ {
        // 这里的状态转移方程就是：求最大和
        // 会面临2种情况，一个是带前面的和，一个是不带前面的和
        dp[i] = max(dp[i-1]+nums[i],nums[i])
        mx = max(mx,dp[i])
    }
    return mx
}

func max(a,b int) int{
    if a>b {
        return a
    }
    return b
}