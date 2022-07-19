package offer

//给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
//每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
//例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
//题意即每段长度和等于绳子长度时，长度之积最大
//设将长度为 n 的绳子切为 a 段：
//n = n_1 + n_2 + ... + n_a
//等价于求解：
//max(n_1 x n_2 x ... x n_a)

//动态规划
func cuttingRope(n int) int {
	dp := make(map[int]int)
	dp[1] = 1
	for i := 2; i < n+1; i++ {
		for j := 1; j < (i/2 + 1); j++ {
			dp[i] = max(dp[i], max(j, dp[j])*max(i-j, dp[i-j]))
		}
	}
	return dp[n]
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

// 递归
func cuttingRope2(n int) int {
	switch n {
	case 2:
		return 1
	case 3:
		return 2
	case 4:
		return 4
	case 5:
		return 6
	case 6:
		return 9
	default:
		return cuttingRope2(n-3) * 3
	}
}
