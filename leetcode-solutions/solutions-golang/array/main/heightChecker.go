package main

import (
	"sort"
)

func heightChecker_s1(heights []int) int {
	target := make([]int, len(heights))
	copy(target, heights)
	res := 0
	sort.Ints(target)
	for i := 0; i < len(heights); i++ {
		if heights[i] != target[i] {
			res++
		}
	}
	return res
}

func heightChecker_s2(heights []int) int {
	target := make([]int, len(heights))
	var diff = []int{}
	copy(target, heights)
	sort.Ints(target)
	for i := 0; i < len(heights); i++ {
		if target[i] != heights[i] {
			diff = append(diff, 1)
		}
	}
	return len(diff)
}
