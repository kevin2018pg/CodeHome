func findLength(nums1 []int, nums2 []int) int {
    dp := make([][]int,len(nums1)+1)
    for i := range dp {
		dp[i] = make([]int, len(nums2)+1)
	}
    res := 0
	for i:=1;i<len(nums1)+1;i++{
	    for j:=1;j<len(nums2)+1;j++{
	        if nums1[i-1] == nums2[j-1] {
	            dp[i][j] = dp[i-1][j-1] + 1
	        }
            res = max(res,dp[i][j])
	    }
	}
    return res
}

func max(a,b int) int {
    if a>b{
        return a
    }
    return b
}