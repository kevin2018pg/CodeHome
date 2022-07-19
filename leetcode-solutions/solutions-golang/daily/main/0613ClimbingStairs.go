package main

//爬楼梯

func climbStairs_s1(n int) int {
	arr := []int{1, 1}
	for i := 2; i <= n; i++ {
		arr = append(arr, arr[i-1]+arr[i-2])
	}
	return arr[n]
}

//最优化dp
func climbStairs_s2(n int) int {
	if n == 1 || n == 2 {
		return n
	}
	a, b := 1, 2
	for i := 3; i < n+1; i++ {
		a, b = b, a+b
	}
	return b
}

//传统dp
func climbStairs_s3(n int) int {
	var tempMap = make(map[int]int)
	tempMap[1] = 1
	tempMap[2] = 2
	for i := 3; i <= n; i++ {
		tempMap[i] = tempMap[i-1] + tempMap[i-2]
	}

	return tempMap[n]
}
