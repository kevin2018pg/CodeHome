// 升序数组平方后再升序排列，需要返回新的数组
// 常规方法：先平方，再排序
// 双指针法：首尾指针

func sortedSquares(nums []int) []int {
    n := len(nums)
    head,tail := 0,n-1
    k := n-1
    res := make([]int,n)
    for head <= tail {
        h2 := nums[head] * nums[head]
        t2 := nums[tail] * nums[tail]
        if h2 < t2{
            res[k] = t2
            tail--
        }else{
            res[k] = h2
            head++
        }
        k--
    }
    return res
}