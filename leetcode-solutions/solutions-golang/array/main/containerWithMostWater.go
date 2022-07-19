package main

func maxArea(height []int) int {
	var i, res int
	j := len(height) - 1
	for i < j {
		if height[i] < height[j] {
			res = max(res, height[i]*(j-i))
			i++
		} else {
			res = max(res, height[j]*(j-i))
			j--
		}
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	h := []int{1, 8, 6, 2, 5, 4, 8, 3, 7}
	maxArea(h)
}
