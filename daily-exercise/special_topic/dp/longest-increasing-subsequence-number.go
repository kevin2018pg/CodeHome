func findNumberOfLIS(nums []int) int {
    n := len(nums)
    if n == 1{
        return n
    }
    dpLength := make([]int,n)
    dpCount := make([]int,n)
    for i:=0;i<n;i++{
        dpLength[i] = 1
        dpCount[i] = 1
    }
    maxLength := 1
    for i:=1;i<n;i++{
        for j:=0;j<i;j++{
            if nums[i] > nums[j] {
                if dpLength[j] + 1 > dpLength[i] {
                    dpLength[i] = dpLength[j]+1
                    dpCount[i] = dpCount[j]
                } else if dpLength[j] + 1 == dpLength[i] {
                    dpCount[i] += dpCount[j]
                }
            }
        }
        maxLength = max(maxLength,dpLength[i])
    }
    maxCount := 0
    for i:=0;i<n;i++{
        if dpLength[i] == maxLength{
            maxCount += dpCount[i]
        }
    }
    return maxCount
}

func max(a,b int) int {
    if a>b{
        return a
    }
    return b
}