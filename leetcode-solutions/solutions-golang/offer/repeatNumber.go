package offer

import "sort"

//线排序之后进行相邻元素之间的比较
func findRepeatNumber_s1(nums []int) int {
	sort.Ints(nums)
	num := nums[0]
	for i := 1; i < len(nums); i++ {
		if num == nums[i] {
			return num
		} else {
			num = nums[i]
		}

	}
	return -1
}

//利用哈希表存储元素
func findRepeatNumber_s2(nums []int) int {
	maps := make(map[int]bool)
	for _, num := range nums {
		if maps[num] {
			return num
		} else {
			maps[num] = true
		}
	}
	return -1
}
