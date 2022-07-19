package offer

//题目同剪绳子1，答案需要取模 1e9+7（1000000007），大数越界情况下的求余问题，直接用动态规划会报错

// 递归
func cuttingRope3(n int) int {
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
		return cuttingRope3(n-3) * 3 % 1000000007
	}
}
