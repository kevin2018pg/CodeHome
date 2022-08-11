func minSubArrayLen(target int, nums []int) int {
    n := len(nums)
    if nums == nil || n == 0{
        return 0
    }
    slow,sumNum := 0,0
    minLength := n+1
    for fast := 0;fast<n;fast++{
        sumNum += nums[fast]
        for sumNum >= target{
            minLength = min(minLength,fast-slow+1)
            sumNum -= nums[slow]
            slow++
        }
    }
    if minLength == n+1{
        return 0
    }else{
        return minLength
    }
}

func min(x,y int) int {
    if x > y{
        return y
    }
    return x
}
