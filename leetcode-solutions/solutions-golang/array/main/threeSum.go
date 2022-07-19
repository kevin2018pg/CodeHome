package main

import "sort"

func threeSum(nums []int) [][]int {
	n := len(nums)
	if nums == nil || n < 3 {
		return nil
	}
	sort.Ints(nums)
	res := [][]int{}
	for k := 0; k < n-2; k++ {
		if nums[k] > 0 {
			break
		}
		if k > 0 && nums[k] == nums[k-1] {
			continue
		}
		i, j := k+1, n-1
		for i < j {
			s := nums[k] + nums[i] + nums[j]
			if s < 0 {
				i++
				for i < j && nums[i] == nums[i-1] {
					i++
				}
			} else if s > 0 {
				j--
				for i < j && nums[j+1] == nums[j] {
					j--
				}
			} else {
				res = append(res, []int{nums[k], nums[i], nums[j]})
				i++
				j--
				for i < j && nums[i] == nums[i-1] {
					i++
				}
				for i < j && nums[j] == nums[j+1] {
					j--
				}
			}
		}
	}
	return res
}
func main() {

}
