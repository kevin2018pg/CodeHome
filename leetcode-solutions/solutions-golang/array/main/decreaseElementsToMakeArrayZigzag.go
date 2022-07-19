package main

import "fmt"

func movesToMakeZigzag(nums []int) int {
	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}
	n, ans := len(nums), []int{0, 0}
	for k := range ans {
		for i := k; i < n; i += 2 {
			d := 0
			if i > 0 {
				d = max(d, nums[i]-nums[i-1]+1)
			}
			if i+1 < n {
				d = max(d, nums[i]-nums[i+1]+1)
			}
			ans[k] += d
		}
	}
	return -max(-ans[0], -ans[1])
}

func main() {
	c := movesToMakeZigzag([]int{9, 6, 1, 6, 2})
	fmt.Println(c)
}
