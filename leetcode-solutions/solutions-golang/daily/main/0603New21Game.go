package main

func new21Game(N int, K int, W int) float64 {
	dp := make([]float64, K+W)
	s := 0.0
	for i := K; i < K+W; i++ {
		if i <= N {
			dp[i] = 1.0
		} else {
			dp[i] = 0.0
		}
		s += dp[i]
	}
	for i := K - 1; i > -1; i-- {
		dp[i] = s / float64(W)
		s = s - dp[i+W] + dp[i]
	}
	return dp[0]
}
