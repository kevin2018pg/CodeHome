func lengthOfLIS(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	dp := make([]int, len(nums))
	result := 1
	for i := 0; i < len(nums); i++ {
		dp[i] = 1   // dp初始状态：都为1，代表每个元素都可以是单独的子序列
		for j := 0; j < i; j++ {    // j 属于[0,i-1]
			if nums[j] < nums[i] {   // 满足递增条件：如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
				dp[i] = max(dp[j]+1, dp[i]) // 转移方程：dp[i]的值代表nums以nums[i]结尾的最长子序列长度
			}
		}
		result = max(result, dp[i]) // 存储更新最大值
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
