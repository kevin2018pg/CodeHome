// 最大子数组和
// 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
// 子数组 是数组中的一个连续部分。
// 1、先排序，直接求正数部分和即可
// 2、状态传递

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