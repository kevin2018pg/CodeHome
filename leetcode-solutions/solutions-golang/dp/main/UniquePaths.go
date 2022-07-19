package main

func uniquePaths_s1(m int, n int) int {
	dp := make([]int, m)
	for i := 0; i < m; i++ {
		dp[i] = 1
	}
	for j := 1; j < n; j++ {
		for i := 1; i < m; i++ {
			dp[i] += dp[i-1]
		}
	}
	return dp[m-1]
}

func uniquePaths_s2(m int, n int) int {
	if m <= 0 || n <= 0 {
		return 0
	}
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}

	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}
