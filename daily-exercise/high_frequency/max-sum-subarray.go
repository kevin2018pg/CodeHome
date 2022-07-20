// 最大子数组和
// 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
// 子数组 是数组中的一个连续部分。
// 1、先排序，遍历至第一个正数累加，如果最后一位是负数，那就返回最后一位
// 2、贪心-状态传递：f(i)=max{f(i−1)+nums[i],nums[i]}

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