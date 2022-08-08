func findLengthOfLCIS(nums []int) int {
    if len(nums) <= 1{
        return len(nums)
    }
    dp := make([]int,len(nums))
    for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
    res := 1
    for i:=1;i<len(nums);i++{
        if nums[i] > nums[i-1]{
            dp[i] = dp[i-1] + 1
        }
        res = max(res, dp[i])
    }
    return res
}

func max(a,b int) int {
    if a > b{
        return a
    }
    return b
}