package main

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func longestConsecutive(nums []int) int {
	numSet := map[int]bool{}
	for _, num := range nums {
		numSet[num] = true
	}
	ans := 0
	for num := range numSet {
		if numSet[num-1] {
			continue
		}
		curlen := 1
		for numSet[num+1] {
			num++
			curlen++
		}
		ans = max(ans, curlen)
	}
	return ans
}
